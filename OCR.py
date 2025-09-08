# OCR_docTR.py
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import re
import tempfile
from PIL import Image

# Bounding box table extraction

# Configuration flags
USE_DYNAMIC_ROWS = True  # Set to False to use fixed row boundaries
DYNAMIC_ROW_TOLERANCE_PIXELS = 15  # Pixel tolerance for grouping rows (will be normalized per image)

def group_by_rows(words, y_tolerance=0.02):
    """
    Dynamic row detection: cluster words by their y_center using tolerance.
    This approach adapts to different image sizes and table layouts.
    Note: y_tolerance should be normalized (0.0-1.0) to match OCR coordinate system.
    """
    if not words:
        print("   ⚠️ No words provided for row grouping")
        return []
    
    print(f"   🔍 Grouping {len(words)} words with normalized y_tolerance={y_tolerance}")
    
    words = sorted(words, key=lambda w: w['y_center'])
    rows, current_row = [], []
    last_y = None
    
    # Debug: show y_center range
    y_centers = [w['y_center'] for w in words]
    print(f"   📏 Y-center range: {min(y_centers):.3f} to {max(y_centers):.3f} (normalized)")
    
    for i, w in enumerate(words):
        if last_y is None or abs(w['y_center'] - last_y) <= y_tolerance:
            current_row.append(w)
        else:
            if current_row:  # Only append non-empty rows
                rows.append(current_row)
                print(f"   📝 Row {len(rows)}: {len(current_row)} words, y_center ≈ {last_y:.3f}")
            current_row = [w]
        last_y = w['y_center']
    
    if current_row:
        rows.append(current_row)
        print(f"   📝 Row {len(rows)}: {len(current_row)} words, y_center ≈ {last_y:.3f}")
    
    print(f"   ✅ Created {len(rows)} rows total")
    return rows

def group_by_fixed_rows(words, row_boundaries):
    """
    Fixed row detection: use predefined normalized row boundaries.
    This is the original approach, kept for compatibility.
    """
    # Assign each word to a row index based on y_center
    rows = [[] for _ in range(len(row_boundaries)-1)]
    for w in words:
        for i in range(len(row_boundaries)-1):
            if row_boundaries[i] <= w['y_center'] < row_boundaries[i+1]:
                rows[i].append(w)
                break
    # Remove empty rows
    rows = [row for row in rows if row]
    return rows

def clean_dataframe(df):
    """Clean the OCR dataframe data"""
    df = df.copy()
    
    # Clean Change column (index 2)
    def clean_change(value):
        if pd.isna(value) or value == '' or value.strip() == '':
            return ''
        value = str(value).strip()
        # Replace I with 1
        value = value.replace('I', '1')
        # Check for NEW
        if value.upper() == 'NEW':
            return 'NEW'
        # Check for + or - followed by numbers
        if re.match(r'^[+-]\d+$', value):
            return value
        # Try to extract number and add sign
        numbers = re.findall(r'\d+', value)
        if numbers:
            if value.startswith('-') or 'down' in value.lower():
                return f"-{numbers[0]}"
            elif value.startswith('+') or 'up' in value.lower():
                return f"+{numbers[0]}"
            else:
                return f"+{numbers[0]}"
        return ''
    
    # Clean numeric columns with K/M suffixes
    def clean_numeric(value):
        if pd.isna(value) or value == '' or value.strip() == '':
            return ''
        value = str(value).strip().replace(',', '').replace(' ', '')
        
        # Handle common OCR misreads first
        ocr_fixes = {
            'O': '0',  # Letter O to zero
            'I': '1',  # Letter I to one
            'l': '1',  # Lowercase L to one
            'S': '5',  # Letter S to five (sometimes)
            'B': '8',  # Letter B to eight (sometimes)
            'Im': '1m',  # Special case for Im -> 1m
            'Ik': '1k',  # Special case for Ik -> 1k
            'lm': '1m',  # Special case for lm -> 1m
            'lk': '1k',  # Special case for lk -> 1k
        }
        
        # Apply OCR fixes
        for wrong, correct in ocr_fixes.items():
            value = value.replace(wrong, correct)
        
        # Ensure k/K and m/M are preceded by numeric values
        # Handle K/k (thousands)
        if value.endswith('k') or value.endswith('K'):
            numeric_part = value[:-1]
            try:
                # Check if it's a valid number
                num = float(numeric_part)
                return str(int(num * 1000))
            except ValueError:
                # If not a valid number, try to extract just digits and decimal
                numeric_match = re.search(r'(\d+\.?\d*)', numeric_part)
                if numeric_match:
                    try:
                        num = float(numeric_match.group(1))
                        return str(int(num * 1000))
                    except:
                        pass
                return value
        
        # Handle M/m (millions)
        elif value.endswith('m') or value.endswith('M'):
            numeric_part = value[:-1]
            try:
                # Check if it's a valid number
                num = float(numeric_part)
                return str(int(num * 1000000))
            except ValueError:
                # If not a valid number, try to extract just digits and decimal
                numeric_match = re.search(r'(\d+\.?\d*)', numeric_part)
                if numeric_match:
                    try:
                        num = float(numeric_match.group(1))
                        return str(int(num * 1000000))
                    except:
                        pass
                return value
        
        # Handle any remaining non-numeric characters at the end
        elif re.search(r'\d+[a-zA-Z]$', value):
            # Extract just the numbers if there are trailing letters
            numbers = re.findall(r'\d+\.?\d*', value)
            if numbers:
                return numbers[0]
        
        # Final cleanup: remove any remaining non-numeric characters except decimal points
        cleaned_value = re.sub(r'[^\d.]', '', value)
        return cleaned_value if cleaned_value else value
    
    # Apply cleaning to Change column
    df['Change'] = df['Change'].apply(clean_change)
    
    # Apply cleaning to Rank and other numeric columns
    numeric_columns = ['Rank', 'Sales', 'Sales %', 'Streams', 'Streams %', 'Airplay', 'Airplay %', 'Units']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    return df

def ocr_boxes_to_dataframe(ocr_results, csv_path, column_boundaries=None, image_path=None):
    # ocr_results: list of dicts with 'text', 'geometry' (bounding box)
    print(f"   📊 Processing {len(ocr_results)} OCR results")
    
    # Calculate normalized tolerance based on image height
    normalized_tolerance = 0.02  # Default fallback
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                image_height = img.height
                normalized_tolerance = DYNAMIC_ROW_TOLERANCE_PIXELS / image_height
                print(f"   📐 Image height: {image_height}px, normalized tolerance: {normalized_tolerance:.4f}")
        except Exception as e:
            print(f"   ⚠️ Could not read image dimensions from {image_path}: {e}")
            print(f"   📐 Using default normalized tolerance: {normalized_tolerance}")
    else:
        print(f"   📐 No image path provided, using default normalized tolerance: {normalized_tolerance}")
    
    # Step 1: Calculate y center for each word
    for w in ocr_results:
        w['y_center'] = (w['geometry'][0][1] + w['geometry'][1][1]) / 2
        w['x_center'] = (w['geometry'][0][0] + w['geometry'][1][0]) / 2

    # Debug: Show some sample words
    if ocr_results:
        print(f"   🔍 Sample words:")
        for i, w in enumerate(ocr_results[:5]):
            print(f"     {i+1}. '{w['text']}' at y={w['y_center']:.3f}, x={w['x_center']:.3f}")
        if len(ocr_results) > 5:
            print(f"     ... and {len(ocr_results) - 5} more words")

    # If no boundaries provided, auto-generate 14 evenly spaced columns
    if column_boundaries is None:
        x_centers = sorted([w['x_center'] for w in ocr_results])
        min_x, max_x = min(x_centers), max(x_centers)
        column_boundaries = [min_x + (max_x - min_x) * i / 14 for i in range(15)]  # 15 edges for 14 columns

    # Step 2: Choose row detection method
    if USE_DYNAMIC_ROWS:
        print(f"   🔄 Using dynamic row detection (tolerance: {DYNAMIC_ROW_TOLERANCE_PIXELS}px = {normalized_tolerance:.4f} normalized)")
        rows = group_by_rows(ocr_results, y_tolerance=normalized_tolerance)
        
        # Fallback: if dynamic detection fails to find rows, use fixed boundaries
        if not rows:
            print("   ⚠️ Dynamic row detection found no rows, falling back to fixed boundaries")
            row_boundaries = [
                0.025, 0.046, 0.083, 0.121, 0.160, 0.195, 0.235, 0.271, 0.306, 0.342,
                0.384, 0.419, 0.454, 0.489, 0.531, 0.565, 0.605, 0.639, 0.677, 0.716,
                0.752, 0.789, 0.827, 0.862, 0.900, 0.937, 0.972
            ]
            rows = group_by_fixed_rows(ocr_results, row_boundaries)
    else:
        print("   📏 Using fixed row boundaries")
        # Use normalized row boundaries for row assignment (original approach)
        row_boundaries = [
            0.025, 0.046, 0.083, 0.121, 0.160, 0.195, 0.235, 0.271, 0.306, 0.342,
            0.384, 0.419, 0.454, 0.489, 0.531, 0.565, 0.605, 0.639, 0.677, 0.716,
            0.752, 0.789, 0.827, 0.862, 0.900, 0.937, 0.972
        ]
        rows = group_by_fixed_rows(ocr_results, row_boundaries)

    print(f"   📊 Detected {len(rows)} rows")

    # Step 3: For each row, assign words to columns
    table = []
    for row_idx, row in enumerate(rows):
        cols = [''] * 15  # One extra column for split Song/Artist
        song_name_words = []
        artist_name_words = []
        
        # Find min/max y in this row for normalization
        min_y = None
        max_y = None
        if row:
            min_y = min(w['geometry'][0][1] for w in row)
            max_y = max(w['geometry'][1][1] for w in row)
            
        for w in row:
            for i in range(14):
                if column_boundaries[i] <= w['x_center'] < column_boundaries[i+1]:
                    if i == 2:
                        # Normalize y_center within row for Song/Artist separation
                        if min_y is not None and max_y is not None:
                            rel_y = (w['y_center'] - min_y) / (max_y - min_y) if max_y > min_y else 0
                            if rel_y <= 0.6:
                                song_name_words.append(w['text'])
                            else:
                                artist_name_words.append(w['text'])
                        else:
                            # Fallback: add to song if we can't determine position
                            song_name_words.append(w['text'])
                    else:
                        # Shift columns after Song/Artist by 1
                        idx = i if i < 2 else i+1
                        cols[idx] = w['text']
                    break
        # Set Song and Artist columns
        cols[2] = ' '.join(song_name_words)
        cols[3] = ' '.join(artist_name_words)
        table.append(cols)

    # Step 4: Build DataFrame with new columns
    columns = [
        'Chart_Date', 'Rank', 'Change', 'Song', 'Artist', 'Points', 'Percent', 'Peak', 'WoC',
        'Sales', 'Sales %', 'Streams', 'Streams %', 'Airplay', 'Airplay %', 'Units'
    ]
    # Extract date from filename (expects csv_path like ocr_table_2025-08-10_1.csv)
    import re
    m = re.search(r'ocr_table_(\d{4})-(\d{2})-(\d{2})_\d+', csv_path)
    if m:
        date_str = f"{m.group(2)}/{m.group(3)}/{m.group(1)[2:]}"  # mm/dd/yy
    else:
        date_str = ''
    # Insert date as first column in each row
    table_with_date = [[date_str] + row for row in table]
    df = pd.DataFrame(table_with_date, columns=columns)
    # Remove the first row if redundant
    df = df.iloc[1:].reset_index(drop=True)
    # # Step 5: Write to CSV
    # df.to_csv(csv_path, index=False)
    # print(f"OCR table written to {csv_path}")
    return df

if __name__ == "__main__":
    DOWNLOAD_DIR = "sorted_images/Aug 25"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Load OCR model
    model = ocr_predictor(pretrained=True)

    # Group images by date
    date_groups = {}
    for filename in os.listdir(DOWNLOAD_DIR):
        if not filename.endswith('.jpg'):
            continue
        # Extract date from filename (e.g., 2025-08-10_1.jpg -> 2025-08-10)
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})_\d+\.jpg', filename)
        if date_match:
            date = date_match.group(1)
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(filename)

    # Process each date group
    all_date_dataframes = []
    temp_files = []  # Track temporary files for cleanup
    
    for date, filenames in date_groups.items():
        print(f"\nProcessing date: {date}")
        date_dataframes = []
        
        # Sort filenames to ensure consistent order (_1, _2, _3, _4)
        filenames.sort()
        
        for filename in filenames:
            image_path = os.path.join(DOWNLOAD_DIR, filename)
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
            
            # Use bounding box approach to build DataFrame and write to CSV
            csv_path = f"ocr_table_{filename.replace('.jpg', '')}.csv"

            # Example: manually set normalized x boundaries for 14 columns
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
            
            # Add to date dataframes
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
            
            # Drop helper columns if they exist
            columns_to_drop = []
            if 'Rank_numeric' in merged_date_df.columns:
                columns_to_drop.append('Rank_numeric')
            if 'original_position' in merged_date_df.columns:
                columns_to_drop.append('original_position')
            
            if columns_to_drop:
                merged_date_df = merged_date_df.drop(columns_to_drop, axis=1)
            
            merged_date_df = merged_date_df.reset_index(drop=True)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            
            merged_date_df.to_csv(temp_file_path, index=False)
            temp_files.append(temp_file_path)
            print(f"Temporary merged dataframe for {date} created")
            
            all_date_dataframes.append((date, merged_date_df))

    # Sort dates in reverse chronological order (most recent first)
    all_date_dataframes.sort(key=lambda x: x[0], reverse=True)
    
    # Merge all dates into final dataframe
    if all_date_dataframes:
        final_df = pd.concat([df for date, df in all_date_dataframes], ignore_index=True)
        final_csv_path = "final_merged_all_dates.csv"
        final_df.to_csv(final_csv_path, index=False)
        print(f"\nFinal merged dataframe written to {final_csv_path}")
        print(f"Total rows: {len(final_df)}")
        print(f"Dates processed: {[date for date, _ in all_date_dataframes]}")
    else:
        print("No valid images found to process")
    
    # Clean up temporary files
    for temp_file_path in temp_files:
        try:
            os.unlink(temp_file_path)
            print(f"Deleted temporary file: {temp_file_path}")
        except Exception as e:
            print(f"Error deleting temporary file {temp_file_path}: {e}")