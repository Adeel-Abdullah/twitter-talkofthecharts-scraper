# OCR_batch_process_mp.py - Multiprocessing version for faster OCR processing
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import re
import tempfile
from multiprocessing import Pool, cpu_count
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Import functions from existing OCR.py
from OCR import clean_dataframe, ocr_boxes_to_dataframe

def process_single_image(args):
    """Process a single image file - designed for multiprocessing"""
    image_path, csv_path, column_boundaries = args
    
    try:
        print(f"[Worker] Processing {image_path}")
        
        # Load OCR model (each process needs its own model instance)
        model = ocr_predictor(pretrained=True)
        
        # Run OCR
        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        json_output = result.export()
        
        # Collect all word boxes from all pages
        ocr_results = []
        for page in json_output["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        ocr_results.append({
                            'text': word["value"],
                            'geometry': word["geometry"]
                        })
        
        # Process OCR results into dataframe
        df = ocr_boxes_to_dataframe(ocr_results, csv_path, column_boundaries=column_boundaries, image_path=image_path)
        # Clean the dataframe
        df = clean_dataframe(df)
        
        print(f"[Worker] Completed {image_path}, got {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"[Worker] Error processing {image_path}: {e}")
        return None

def process_date_group_mp(date, filenames, input_dir, max_workers=None):
    """Process a group of images for a specific date using multiprocessing"""
    print(f"\n[MP] Processing date: {date} with {len(filenames)} files")
    
    # Use the normalized column boundaries (matching OCR.py)
    column_boundaries = [
        0.021,  # left edge of Rank
        0.067,  # left edge of Change
        0.110,  # left edge of Song
        0.434,  # left edge of Artist
        0.491,  # left edge of Points
        0.536,  # left edge of Percent
        0.572,  # left edge of Peak
        0.613,  # left edge of WoC
        0.667,  # left edge of Sales
        0.706,  # left edge of Sales %
        0.767,  # left edge of Streams
        0.806,  # left edge of Streams %
        0.869,  # left edge of Airplay
        0.908,  # left edge of Airplay %
        0.974   # right edge of Units
    ]
    
    # Prepare arguments for multiprocessing
    process_args = []
    for filename in filenames:
        image_path = os.path.join(input_dir, filename)
        csv_path = f"ocr_table_{filename.replace('.jpg', '.csv')}"
        process_args.append((image_path, csv_path, column_boundaries))
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(filenames))
    
    print(f"[MP] Using {max_workers} workers for {len(filenames)} files")
    
    # Process files in parallel
    dataframes = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_filename = {
            executor.submit(process_single_image, args): args[0] 
            for args in process_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_filename):
            image_path = future_to_filename[future]
            try:
                df = future.result()
                if df is not None:
                    dataframes.append(df)
                else:
                    print(f"[MP] Failed to process {image_path}")
            except Exception as e:
                print(f"[MP] Exception processing {image_path}: {e}")
    
    end_time = time.time()
    print(f"[MP] Completed {date} in {end_time - start_time:.2f} seconds")
    
    return dataframes

def process_directory_mp(input_dir, output_csv_name, max_workers=None):
    """Process all images in a directory using multiprocessing and return merged dataframe"""
    
    # Group images by date
    date_groups = {}
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jpg'):
            continue
        # Extract date from filename (e.g., 2025-08-10_1.jpg -> 2025-08-10)
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})_\d+\.jpg', filename)
        if date_match:
            date = date_match.group(1)
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(filename)

    if not date_groups:
        print(f"[MP] No valid images found in {input_dir}")
        return None

    print(f"[MP] Found {len(date_groups)} date groups with total {sum(len(files) for files in date_groups.values())} images")
    
    # Process each date group
    all_date_dataframes = []
    
    for date, filenames in date_groups.items():
        # Sort filenames to ensure consistent order (_1, _2, _3, _4)
        filenames.sort()
        
        # Process this date group with multiprocessing
        date_dataframes = process_date_group_mp(date, filenames, input_dir, max_workers)
        
        # Merge all dataframes for this date
        if date_dataframes:
            merged_date_df = pd.concat(date_dataframes, ignore_index=True)
            
            # Preserve original position for rows with empty ranks
            merged_date_df['original_position'] = merged_date_df.index
            
            # Convert Rank column to numeric for sorting, handle non-numeric values
            def safe_rank_convert(x):
                try:
                    if x and str(x).strip() != '':
                        return int(x)
                    else:
                        return None  # Keep empty ranks as None instead of 999
                except:
                    return None
            
            merged_date_df['Rank_numeric'] = merged_date_df['Rank'].apply(safe_rank_convert)
            
            # Sort only rows with valid ranks, preserve position of empty rank rows
            rows_with_rank = merged_date_df[merged_date_df['Rank_numeric'].notna()].copy()
            rows_without_rank = merged_date_df[merged_date_df['Rank_numeric'].isna()].copy()
            
            # Sort rows with valid ranks
            rows_with_rank = rows_with_rank.sort_values('Rank_numeric')
            
            # Combine: keep empty rank rows in original positions, sort others by rank
            final_rows = []
            for idx in range(len(merged_date_df)):
                original_row = merged_date_df.iloc[idx]
                if pd.isna(original_row['Rank_numeric']):
                    # Keep empty rank row in original position
                    final_rows.append(original_row)
                else:
                    # This position will be filled by sorted rank rows later
                    final_rows.append(None)
            
            # Fill in the sorted rank rows
            sorted_rank_idx = 0
            for i, row in enumerate(final_rows):
                if row is None and sorted_rank_idx < len(rows_with_rank):
                    final_rows[i] = rows_with_rank.iloc[sorted_rank_idx]
                    sorted_rank_idx += 1
            
            # Convert back to DataFrame
            merged_date_df = pd.DataFrame([row for row in final_rows if row is not None])
            
            # Drop helper columns if they exist
            columns_to_drop = []
            if 'Rank_numeric' in merged_date_df.columns:
                columns_to_drop.append('Rank_numeric')
            if 'original_position' in merged_date_df.columns:
                columns_to_drop.append('original_position')
            
            if columns_to_drop:
                merged_date_df = merged_date_df.drop(columns_to_drop, axis=1)
            
            merged_date_df = merged_date_df.reset_index(drop=True)
            
            print(f"[MP] Merged {len(date_dataframes)} dataframes for {date}")
            all_date_dataframes.append((date, merged_date_df))

    # Sort dates in reverse chronological order (most recent first)
    all_date_dataframes.sort(key=lambda x: x[0], reverse=True)
    
    # Merge all dates into final dataframe
    final_df = None
    if all_date_dataframes:
        final_df = pd.concat([df for date, df in all_date_dataframes], ignore_index=True)
        print(f"[MP] Final merged dataframe created for {input_dir}")
        print(f"[MP] Total rows: {len(final_df)}")
        print(f"[MP] Dates processed: {[date for date, _ in all_date_dataframes]}")
    
    return final_df

def process_subdirectory_wrapper(args):
    """Wrapper function for multiprocessing subdirectory processing"""
    subdir, sorted_images_dir, chart_dfs_dir, max_workers_per_subdir = args
    
    print(f"\n[MP-SUB] {'='*50}")
    print(f"[MP-SUB] Processing subdirectory: {subdir}")
    print(f"[MP-SUB] {'='*50}")
    
    subdir_path = os.path.join(sorted_images_dir, subdir)
    output_csv_name = f"{subdir.replace(' ', '_')}_merged.csv"
    
    # Process the subdirectory
    start_time = time.time()
    final_df = process_directory_mp(subdir_path, output_csv_name, max_workers_per_subdir)
    end_time = time.time()
    
    if final_df is not None:
        # Save to subdirectory
        subdir_csv_path = os.path.join(subdir_path, output_csv_name)
        final_df.to_csv(subdir_csv_path, index=False)
        print(f"[MP-SUB] CSV saved to subdirectory: {subdir_csv_path}")
        
        # Save to chart_dfs directory
        chart_dfs_csv_path = os.path.join(chart_dfs_dir, output_csv_name)
        final_df.to_csv(chart_dfs_csv_path, index=False)
        print(f"[MP-SUB] CSV saved to chart_dfs: {chart_dfs_csv_path}")
        
        print(f"[MP-SUB] Completed {subdir} in {end_time - start_time:.2f} seconds")
        return (f"SUCCESS: {subdir} - {len(final_df)} rows", final_df)
    else:
        print(f"[MP-SUB] No data processed for subdirectory: {subdir}")
        return (f"FAILED: {subdir}", None)

def merge_all_csvs_in_chart_dfs(chart_dfs_dir):
    """Merge all CSV files in chart_dfs directory in correct order"""
    print(f"\n[MERGE] Starting merge of all CSV files in {chart_dfs_dir}")
    
    # Get all CSV files in chart_dfs directory
    csv_files = [f for f in os.listdir(chart_dfs_dir) if f.endswith('.csv') and f != 'final_merged_all_dates.csv']
    
    if not csv_files:
        print(f"[MERGE] No CSV files found in {chart_dfs_dir}")
        return None
    
    print(f"[MERGE] Found {len(csv_files)} CSV files to merge: {csv_files}")
    
    # Load all CSV files
    all_dataframes = []
    for csv_file in csv_files:
        csv_path = os.path.join(chart_dfs_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                all_dataframes.append(df)
                print(f"[MERGE] Loaded {csv_file}: {len(df)} rows")
            else:
                print(f"[MERGE] Skipped empty file: {csv_file}")
        except Exception as e:
            print(f"[MERGE] Error loading {csv_file}: {e}")
    
    if not all_dataframes:
        print(f"[MERGE] No valid data found in CSV files")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"[MERGE] Combined {len(all_dataframes)} files into {len(combined_df)} total rows")
    
    # Convert Chart_Date to datetime for proper sorting
    try:
        # Try different date formats
        combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], format='%d-%m-%y', errors='coerce')
        if combined_df['Chart_Date_parsed'].isna().all():
            combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], format='%Y-%m-%d', errors='coerce')
        if combined_df['Chart_Date_parsed'].isna().all():
            combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], errors='coerce')
    except Exception as e:
        print(f"[MERGE] Warning: Could not parse dates properly: {e}")
        # Fallback: sort by Chart_Date as string
        combined_df['Chart_Date_parsed'] = combined_df['Chart_Date']
    
    # Convert Rank to numeric for sorting
    def safe_rank_convert(x):
        try:
            if x and str(x).strip() != '':
                return int(x)
            else:
                return float('inf')  # Put empty ranks at the end
        except:
            return float('inf')
    
    combined_df['Rank_numeric'] = combined_df['Rank'].apply(safe_rank_convert)
    
    # Sort by Chart_Date (descending - most recent first) then by Rank (ascending)
    combined_df = combined_df.sort_values(['Chart_Date_parsed', 'Rank_numeric'], ascending=[False, True])
    
    # Drop helper columns
    final_df = combined_df.drop(['Chart_Date_parsed', 'Rank_numeric'], axis=1).reset_index(drop=True)
    
    print(f"[MERGE] Final merged dataframe: {len(final_df)} rows")
    print(f"[MERGE] Date range: {final_df['Chart_Date'].min()} to {final_df['Chart_Date'].max()}")
    
    return final_df

if __name__ == "__main__":
    SORTED_IMAGES_DIR = "sorted_images"
    CHART_DFS_DIR = "chart_dfs"
    
    # Multiprocessing configuration
    MAX_WORKERS_PER_SUBDIR = 2  # Workers per subdirectory (for image processing)
    MAX_SUBDIRS_PARALLEL = 2   # Number of subdirectories to process in parallel
    
    print(f"[MP] Starting multiprocessing OCR batch processor")
    print(f"[MP] Available CPU cores: {cpu_count()}")
    print(f"[MP] Max workers per subdirectory: {MAX_WORKERS_PER_SUBDIR}")
    print(f"[MP] Max subdirectories in parallel: {MAX_SUBDIRS_PARALLEL}")
    
    # Create chart_dfs directory if it doesn't exist
    os.makedirs(CHART_DFS_DIR, exist_ok=True)
    
    # Check if sorted_images directory exists
    if not os.path.exists(SORTED_IMAGES_DIR):
        print(f"[MP] Directory {SORTED_IMAGES_DIR} not found. Processing default charts_images directory instead.")
        DOWNLOAD_DIR = "charts_images"
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
        # Process the default directory
        start_time = time.time()
        final_df = process_directory_mp(DOWNLOAD_DIR, "final_merged_all_dates.csv", MAX_WORKERS_PER_SUBDIR)
        end_time = time.time()
        
        if final_df is not None:
            # Save individual CSV to chart_dfs
            chart_dfs_path = os.path.join(CHART_DFS_DIR, "charts_images_merged.csv")
            final_df.to_csv(chart_dfs_path, index=False)
            print(f"[MP] Processed data saved to {chart_dfs_path}")
            print(f"[MP] Processing time: {end_time - start_time:.2f} seconds")
    else:
        # Process each subdirectory in sorted_images
        subdirs = [d for d in os.listdir(SORTED_IMAGES_DIR) 
                  if os.path.isdir(os.path.join(SORTED_IMAGES_DIR, d))]
        
        if not subdirs:
            print(f"[MP] No subdirectories found in {SORTED_IMAGES_DIR}")
        else:
            print(f"[MP] Found {len(subdirs)} subdirectories to process: {subdirs}")
            
            # Prepare arguments for subdirectory processing
            subdir_args = [
                (subdir, SORTED_IMAGES_DIR, CHART_DFS_DIR, MAX_WORKERS_PER_SUBDIR)
                for subdir in subdirs
            ]
            
            # Process subdirectories in parallel
            start_time = time.time()
            
            if len(subdirs) == 1 or MAX_SUBDIRS_PARALLEL == 1:
                # Process sequentially if only one subdirectory or parallel disabled
                results = []
                for args in subdir_args:
                    result = process_subdirectory_wrapper(args)
                    results.append(result)
            else:
                # Process multiple subdirectories in parallel
                with ProcessPoolExecutor(max_workers=MAX_SUBDIRS_PARALLEL) as executor:
                    results = list(executor.map(process_subdirectory_wrapper, subdir_args))
            
            end_time = time.time()
            
            # Show processing results
            print(f"\n[MP] {'='*50}")
            print("[MP] Processing complete for all subdirectories!")
            print(f"[MP] Total processing time: {end_time - start_time:.2f} seconds")
            print("[MP] Results:")
            for result in results:
                message, df = result
                print(f"[MP]   {message}")
            print(f"[MP] {'='*50}")
    
    # Final merge step: Merge all CSVs in chart_dfs directory
    print(f"\n[MP] Starting final merge of all CSV files...")
    final_merge_start = time.time()
    
    final_complete_df = merge_all_csvs_in_chart_dfs(CHART_DFS_DIR)
    
    if final_complete_df is not None:
        # Save final complete merged CSV
        final_csv_path = "final_merged_all_dates.csv"
        final_complete_df.to_csv(final_csv_path, index=False)
        print(f"[MERGE] Final complete merged CSV written to {final_csv_path}")
        
        # Also save to chart_dfs directory
        chart_dfs_final_path = os.path.join(CHART_DFS_DIR, "final_merged_all_dates.csv")
        final_complete_df.to_csv(chart_dfs_final_path, index=False)
        print(f"[MERGE] Final complete merged CSV also saved to {chart_dfs_final_path}")
        
        final_merge_end = time.time()
        print(f"[MERGE] Final merge completed in {final_merge_end - final_merge_start:.2f} seconds")
        print(f"[MERGE] Total rows in final CSV: {len(final_complete_df)}")
        
        # Show sample of final data
        print(f"\n[MERGE] Sample of final merged data:")
        print(final_complete_df[['Chart_Date', 'Rank', 'Song', 'Artist']].head(10).to_string(index=False))
    else:
        print(f"[MERGE] No data available for final merge")
