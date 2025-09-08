# OCR_batch_process.py
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import re
import tempfile
import time

# Import functions from existing OCR.py
from OCR import clean_dataframe, ocr_boxes_to_dataframe

def process_directory(input_dir, output_csv_name):
    """Process all images in a directory and return merged dataframe"""
    # Load OCR model
    model = ocr_predictor(pretrained=True)

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
        print(f"No valid images found in {input_dir}")
        return None

    # Process each date group
    all_date_dataframes = []
    temp_files = []  # Track temporary files for cleanup
    
    for date, filenames in date_groups.items():
        print(f"\nProcessing date: {date}")
        date_dataframes = []
        
        # Sort filenames to ensure consistent order (_1, _2, _3, _4)
        filenames.sort()
        
        for filename in filenames:
            image_path = os.path.join(input_dir, filename)
            print(f"Processing {image_path}")
            
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
            
            # Create output CSV path
            csv_path = f"ocr_table_{filename.replace('.jpg', '.csv')}"
            
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
            df = ocr_boxes_to_dataframe(ocr_results, csv_path, column_boundaries=column_boundaries, image_path=image_path)
            # Clean the dataframe
            df = clean_dataframe(df)
            
            print(f"Processed {filename}, got {len(df)} rows")
            date_dataframes.append(df)
        
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
            merged_date_df = merged_date_df.drop(['Rank_numeric', 'original_position'], axis=1)
            merged_date_df = merged_date_df.reset_index(drop=True)
            
            print(f"Merged {len(date_dataframes)} dataframes for {date}")
            all_date_dataframes.append((date, merged_date_df))

    # Sort dates in reverse chronological order (most recent first)
    all_date_dataframes.sort(key=lambda x: x[0], reverse=True)
    
    # Merge all dates into final dataframe
    final_df = None
    if all_date_dataframes:
        final_df = pd.concat([df for date, df in all_date_dataframes], ignore_index=True)
        print(f"\nFinal merged dataframe created for {input_dir}")
        print(f"Total rows: {len(final_df)}")
        print(f"Dates processed: {[date for date, _ in all_date_dataframes]}")
    
    return final_df

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
    
    # Create chart_dfs directory if it doesn't exist
    os.makedirs(CHART_DFS_DIR, exist_ok=True)
    
    # Check if sorted_images directory exists
    if not os.path.exists(SORTED_IMAGES_DIR):
        print(f"Directory {SORTED_IMAGES_DIR} not found. Processing default charts_images directory instead.")
        DOWNLOAD_DIR = "charts_images"
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
        # Process the default directory
        final_df = process_directory(DOWNLOAD_DIR, "final_merged_all_dates.csv")
        if final_df is not None:
            # Save individual CSV to chart_dfs
            chart_dfs_path = os.path.join(CHART_DFS_DIR, "charts_images_merged.csv")
            final_df.to_csv(chart_dfs_path, index=False)
            print(f"Processed data saved to {chart_dfs_path}")
    else:
        # Process each subdirectory in sorted_images
        subdirs = [d for d in os.listdir(SORTED_IMAGES_DIR) 
                  if os.path.isdir(os.path.join(SORTED_IMAGES_DIR, d))]
        
        if not subdirs:
            print(f"No subdirectories found in {SORTED_IMAGES_DIR}")
        else:
            print(f"Found {len(subdirs)} subdirectories to process: {subdirs}")
            
            for subdir in subdirs:
                print(f"\n{'='*50}")
                print(f"Processing subdirectory: {subdir}")
                print(f"{'='*50}")
                
                subdir_path = os.path.join(SORTED_IMAGES_DIR, subdir)
                output_csv_name = f"{subdir.replace(' ', '_')}_merged.csv"
                
                # Process the subdirectory
                final_df = process_directory(subdir_path, output_csv_name)
                
                if final_df is not None:
                    # Save to subdirectory
                    subdir_csv_path = os.path.join(subdir_path, output_csv_name)
                    final_df.to_csv(subdir_csv_path, index=False)
                    print(f"CSV saved to subdirectory: {subdir_csv_path}")
                    
                    # Save to chart_dfs directory
                    chart_dfs_csv_path = os.path.join(CHART_DFS_DIR, output_csv_name)
                    final_df.to_csv(chart_dfs_csv_path, index=False)
                    print(f"CSV saved to chart_dfs: {chart_dfs_csv_path}")
                else:
                    print(f"No data processed for subdirectory: {subdir}")
            
            print(f"\n{'='*50}")
            print("Processing complete for all subdirectories!")
            print(f"{'='*50}")
            
            print(f"\n{'='*50}")
            print("Processing complete for all subdirectories!")
            print(f"{'='*50}")
    
    # Final merge step: Merge all CSVs in chart_dfs directory
    print(f"\nStarting final merge of all CSV files...")
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
