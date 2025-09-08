# OCR_batch_process_colab.py - Google Colab optimized version
import pandas as pd
import os
import re
import time

# Add RapidFuzz for fuzzy matching and standardization
try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Levenshtein
    RAPIDFUZZ_AVAILABLE = True
    print("✅ RapidFuzz available for fuzzy standardization")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️ RapidFuzz not available. Install with: !pip install rapidfuzz")

# Install required packages (run this in a separate cell first)
# !pip install doctr[torch] pandas pillow python-dateutil openpyxl rapidfuzz

# Global variables for OCR components
ocr_model = None
DocumentFile = None

def setup_doctr():
    """Setup DocTR imports with proper error handling and GPU configuration"""
    global ocr_model, DocumentFile
    
    try:
        # Import DocTR components
        from doctr.io import DocumentFile as DF
        from doctr.models import ocr_predictor
        DocumentFile = DF
        
        # Check for GPU availability
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize OCR predictor with GPU support
        print(f"🔧 Initializing DocTR with device: {device}")
        if device == 'cuda':
            print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
            # Initialize with CUDA support
            ocr_model = ocr_predictor(pretrained=True, assume_straight_pages=True).cuda()
        else:
            print("💻 Using CPU")
            ocr_model = ocr_predictor(pretrained=True, assume_straight_pages=True)
        
        print("✅ DocTR imported and initialized successfully")
        return True
    except ImportError as e:
        print(f"❌ DocTR import failed: {e}")
        print("Please install doctr: !pip install doctr[torch]")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing DocTR: {e}")
        return False

# Try to import from uploaded OCR.py, otherwise use simplified versions
try:
    # If you uploaded OCR.py to Colab
    import sys
    sys.path.append('/content')
    from OCR import clean_dataframe, ocr_boxes_to_dataframe, DYNAMIC_ROW_TOLERANCE_PIXELS
    print("✅ Successfully imported from OCR.py (should use mm/dd/yy date format)")
    OCR_VERSION = "full"
except ImportError:
    print("⚠️ OCR.py not found. Using simplified versions with mm/dd/yy date format.")
    OCR_VERSION = "simplified"
    
    # Simplified versions for Colab (you may need to upload full OCR.py for best results)
    DYNAMIC_ROW_TOLERANCE_PIXELS = 15
    
    def clean_dataframe(df):
        """Simplified cleaning function - upload OCR.py for full functionality"""
        # Apply text normalization even in simplified mode
        df = normalize_dataframe_text(df)
        return df
    
    def ocr_boxes_to_dataframe(ocr_results, csv_path, column_boundaries=None, image_path=None):
        """Simplified OCR processing - upload OCR.py for full functionality"""
        print("⚠️ Using simplified OCR processing. Upload OCR.py for full functionality.")
        # Create a basic dataframe structure
        columns = ['Chart_Date', 'Rank', 'Change', 'Song', 'Artist', 'Points', 'Percent', 'Peak', 'WoC',
                  'Sales', 'Sales %', 'Streams', 'Streams %', 'Airplay', 'Airplay %', 'Units']
        
        # Extract date from filename (expects csv_path like ocr_table_2025-08-10_1.csv)
        import re
        m = re.search(r'ocr_table_(\d{4})-(\d{2})-(\d{2})_\d+', csv_path)
        if m:
            date_str = f"{m.group(2)}/{m.group(3)}/{m.group(1)[2:]}"  # mm/dd/yy format
        else:
            date_str = ''
        
        # Extract basic text from OCR results
        texts = [result['text'] for result in ocr_results]
        
        # Create simple rows (this is very basic - OCR.py has much better logic)
        rows = []
        for i in range(0, len(texts), 16):  # Assume 16 columns per row
            row = texts[i:i+16]
            if len(row) < 16:
                row.extend([''] * (16 - len(row)))
            rows.append(row)
        
        # Insert date as first column in each row
        table_with_date = [[date_str] + row for row in rows]
        df = pd.DataFrame(table_with_date, columns=columns)
        return df

def normalize_text(text):
    """Normalize text data: lowercase, strip punctuation, remove whitespace, normalize accents"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).strip()
    
    # Remove leading "=" signs
    text = re.sub(r'^=+', '', text)
    
    # Normalize accents (basic mapping)
    accent_map = {
        'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ý': 'y', 'ÿ': 'y',
        'ñ': 'n', 'ç': 'c',
        'Á': 'A', 'À': 'A', 'Â': 'A', 'Ä': 'A', 'Ã': 'A', 'Å': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Ö': 'O', 'Õ': 'O',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ý': 'Y', 'Ÿ': 'Y',
        'Ñ': 'N', 'Ç': 'C'
    }
    
    for accented, normal in accent_map.items():
        text = text.replace(accented, normal)
    
    # Convert to lowercase
    text = text.lower()
    
    # Strip common punctuation (but keep some like apostrophes and periods in names)
    # Remove: []{}()""''«»‚„"" but keep: .-'&
    text = re.sub(r'[[\]{}()""''«»‚„""]+', '', text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def music_specific_normalize(text):
    """Apply music industry specific normalizations"""
    if pd.isna(text) or text == '':
        return text
    
    text = str(text)
    
    # Standardize featuring credits
    text = re.sub(r'\b(feat\.?|featuring|ft\.?)\b', 'feat.', text, flags=re.IGNORECASE)
    
    # Standardize "and" vs "&" 
    text = re.sub(r'\s+&\s+', ' and ', text)
    
    # Remove common parenthetical info like "(Remix)", "(Radio Edit)", "(Live)"
    text = re.sub(r'\s*\([^)]*remix[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\([^)]*radio\s+edit[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\([^)]*live[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\([^)]*acoustic[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\([^)]*instrumental[^)]*\)', '', text, flags=re.IGNORECASE)
    
    # Clean up any double spaces created by removals
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_change_from_text(text, current_change):
    """Extract numeric change values from song/artist text and update change column"""
    if pd.isna(text) or text == '':
        return text, current_change
    
    # Only process if change column is currently empty
    if current_change and str(current_change).strip() != '':
        return text, current_change
    
    text = str(text).strip()
    
    # Pattern for numbers at the beginning: "5 Song Name" or "-3 Artist Name"
    pattern_start = r'^([+-]?\d+)\s+(.+)$'
    match_start = re.match(pattern_start, text)
    
    if match_start:
        number_str, remaining_text = match_start.groups()
        number = int(number_str)
        
        # Format change value
        if number > 0:
            new_change = f"+{number}"
        else:
            new_change = str(number)  # Already has minus sign
        
        return remaining_text.strip(), new_change
    
    # Pattern for negative numbers: "-5 Song Name"
    pattern_negative = r'^-(\d+)\s+(.+)$'
    match_negative = re.match(pattern_negative, text)
    
    if match_negative:
        number_str, remaining_text = match_negative.groups()
        new_change = f"-{number_str}"
        return remaining_text.strip(), new_change
    
    # No numeric values found at the beginning
    return text, current_change

def normalize_dataframe_text(df, enable_fuzzy_standardization=False, fuzzy_threshold=85):
    """Apply comprehensive text normalization to dataframe"""
    if df is None or df.empty:
        return df
    
    print("\n🧹 Applying text normalization...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Text columns to normalize
    text_columns = ['Song', 'Artist']
    
    # First, extract change values from Song and Artist columns if Change is empty
    if 'Change' in df.columns:
        for col in text_columns:
            if col in df.columns:
                print(f"   🔍 Extracting change values from {col} column...")
                
                def process_row(row):
                    text_val = row[col]
                    change_val = row['Change']
                    new_text, new_change = extract_change_from_text(text_val, change_val)
                    return pd.Series([new_text, new_change])
                
                # Apply extraction
                extracted = df.apply(process_row, axis=1)
                df[col] = extracted[0]
                df['Change'] = extracted[1]
    
    # Then normalize all text columns
    for col in text_columns:
        if col in df.columns:
            print(f"   🧽 Normalizing {col} column...")
            original_sample = df[col].dropna().head(3).tolist()
            print(f"      Before: {original_sample}")
            
            # Apply basic text normalization
            df[col] = df[col].apply(normalize_text)
            
            # Apply music-specific normalization
            df[col] = df[col].apply(music_specific_normalize)
            
            normalized_sample = df[col].dropna().head(3).tolist()
            print(f"      After:  {normalized_sample}")
    
    print("✅ Text normalization completed (including music-specific rules)")
    
    # Apply fuzzy standardization if requested
    if enable_fuzzy_standardization and RAPIDFUZZ_AVAILABLE:
        original_unique_combos = len(df.dropna(subset=['Song', 'Artist']).groupby(['Song', 'Artist'])) if 'Song' in df.columns and 'Artist' in df.columns else 0
        df = advanced_standardization(df, debug=True)
        final_unique_combos = len(df.dropna(subset=['Song', 'Artist']).groupby(['Song', 'Artist'])) if 'Song' in df.columns and 'Artist' in df.columns else 0
        print(f"✅ Fuzzy standardization completed: {original_unique_combos} → {final_unique_combos} unique combinations")
    elif enable_fuzzy_standardization and not RAPIDFUZZ_AVAILABLE:
        print("⚠️ Fuzzy standardization requested but RapidFuzz not available")
    
    return df

def standardize_song_artist_combinations(df, similarity_threshold=85, debug=True):
    """
    Standardize song-artist combinations using fuzzy matching without reducing dataset size.
    Similar combinations get the same canonical spelling while preserving all individual rows.
    
    Args:
        df: DataFrame with Song and Artist columns
        similarity_threshold: Minimum similarity score (0-100) to consider items as variations
        debug: Show detailed information about standardizations
    
    Returns:
        DataFrame with standardized song/artist spellings (same number of rows)
    """
    if not RAPIDFUZZ_AVAILABLE:
        print("⚠️ RapidFuzz not available. Skipping fuzzy standardization.")
        print("Install with: !pip install rapidfuzz")
        return df
    
    if df is None or df.empty:
        return df
    
    if 'Song' not in df.columns or 'Artist' not in df.columns:
        print("⚠️ Song or Artist column not found. Skipping fuzzy standardization.")
        return df
    
    print(f"\n🔧 Starting fuzzy standardization (threshold: {similarity_threshold}%)")
    print(f"📊 Input: {len(df)} rows (will remain {len(df)} rows)")
    
    # Create a copy to work with
    df_work = df.copy()
    
    # Remove rows with empty songs/artists for processing
    valid_mask = (df_work['Song'].notna() & df_work['Artist'].notna() & 
                  (df_work['Song'].astype(str).str.strip() != '') & 
                  (df_work['Artist'].astype(str).str.strip() != ''))
    
    df_valid = df_work[valid_mask].copy()
    df_invalid = df_work[~valid_mask].copy()
    
    print(f"📊 Processing {len(df_valid)} rows with valid song-artist data")
    
    if len(df_valid) == 0:
        return df
    
    # Create song-artist combinations for matching
    df_valid['song_artist_combo'] = df_valid['Song'].astype(str) + " | " + df_valid['Artist'].astype(str)
    
    # Find unique combinations and create standardization mapping
    unique_combos = df_valid['song_artist_combo'].unique()
    standardization_map = {}
    combo_groups = []
    processed = set()
    
    print(f"🔍 Analyzing {len(unique_combos)} unique combinations...")
    
    # Group similar combinations
    for i, combo1 in enumerate(unique_combos):
        if combo1 in processed:
            continue
        
        current_group = [combo1]
        processed.add(combo1)
        
        # Find similar combinations
        for j, combo2 in enumerate(unique_combos[i+1:], i+1):
            if combo2 in processed:
                continue
            
            # Calculate similarity
            similarity = fuzz.ratio(combo1, combo2)
            
            if similarity >= similarity_threshold:
                current_group.append(combo2)
                processed.add(combo2)
        
        if len(current_group) > 1:
            combo_groups.append(current_group)
            if debug:
                print(f"🔗 Found variation group (similarity ≥{similarity_threshold}%):")
                for combo in current_group:
                    print(f"   - {combo}")
    
    print(f"📊 Found {len(combo_groups)} groups with variations")
    
    # Create standardization mapping
    standardizations_applied = 0
    
    for group in combo_groups:
        # Choose canonical version (most frequently occurring is most accurate)
        combo_counts = df_valid[df_valid['song_artist_combo'].isin(group)]['song_artist_combo'].value_counts()
        canonical_combo = combo_counts.index[0]  # Most frequent version
        canonical_song, canonical_artist = canonical_combo.split(" | ", 1)
        
        if debug:
            print(f"🎯 Canonical: '{canonical_song}' by '{canonical_artist}' (appears {combo_counts.iloc[0]} times)")
        
        # Map all variations to canonical
        for combo in group:
            if combo != canonical_combo:
                song, artist = combo.split(" | ", 1)
                standardization_map[combo] = (canonical_song, canonical_artist)
                standardizations_applied += 1
                
                if debug:
                    print(f"   📝 '{song}' by '{artist}' → '{canonical_song}' by '{canonical_artist}'")
    
    # Apply standardizations
    for idx, row in df_valid.iterrows():
        combo = row['song_artist_combo']
        if combo in standardization_map:
            canonical_song, canonical_artist = standardization_map[combo]
            df_valid.at[idx, 'Song'] = canonical_song
            df_valid.at[idx, 'Artist'] = canonical_artist
    
    # Clean up helper column
    df_valid = df_valid.drop('song_artist_combo', axis=1)
    
    # Combine valid and invalid rows back together
    if len(df_invalid) > 0:
        result_df = pd.concat([df_valid, df_invalid], ignore_index=True)
    else:
        result_df = df_valid
    
    # Restore original order
    result_df = result_df.sort_index().reset_index(drop=True)
    
    print(f"✅ Standardization completed:")
    print(f"   📊 Total rows: {len(result_df)} (unchanged)")
    print(f"   🔧 Combinations standardized: {standardizations_applied}")
    print(f"   📈 Variation groups processed: {len(combo_groups)}")
    
    return result_df

def standardize_artists(df, threshold=80, debug=True):
    """Standardize artist name variations across the entire dataset"""
    if not RAPIDFUZZ_AVAILABLE or df.empty:
        return df
    
    print(f"\n🎤 Standardizing artist variations (threshold: {threshold}%)")
    
    # Get unique artists
    valid_artists = df[df['Artist'].notna() & (df['Artist'].astype(str).str.strip() != '')]['Artist'].unique()
    artist_mapping = {}
    processed = set()
    standardizations = 0
    
    for artist1 in valid_artists:
        if artist1 in processed:
            continue
        
        # Find similar artists
        similar_artists = [artist1]
        processed.add(artist1)
        
        for artist2 in valid_artists:
            if artist2 in processed:
                continue
            
            similarity = fuzz.ratio(str(artist1), str(artist2))
            if similarity >= threshold:
                similar_artists.append(artist2)
                processed.add(artist2)
        
        if len(similar_artists) > 1:
            # Choose the most frequently occurring version as canonical
            artist_counts = df[df['Artist'].isin(similar_artists)]['Artist'].value_counts()
            canonical = artist_counts.index[0]  # Most frequent version
            
            if debug:
                print(f"🎤 Artist standardization: '{canonical}' ← {len(similar_artists)} variations (appears {artist_counts.iloc[0]} times)")
                for artist in similar_artists:
                    if artist != canonical:
                        print(f"   📝 '{artist}' → '{canonical}'")
            
            for artist in similar_artists:
                if artist != canonical:
                    artist_mapping[artist] = canonical
                    standardizations += 1
    
    # Apply artist mapping
    df_result = df.copy()
    df_result['Artist'] = df_result['Artist'].map(artist_mapping).fillna(df_result['Artist'])
    
    print(f"✅ Artist standardization: {standardizations} changes applied")
    return df_result

def standardize_songs_by_artist(df, threshold=85, debug=True):
    """Standardize song title variations within same artist"""
    if not RAPIDFUZZ_AVAILABLE or df.empty:
        return df
    
    print(f"\n🎵 Standardizing song variations by artist (threshold: {threshold}%)")
    
    result_df = df.copy()
    total_standardizations = 0
    
    # Group by artist
    for artist, artist_group in df.groupby('Artist'):
        if len(artist_group) <= 1:
            continue
        
        # Get unique songs for this artist
        valid_songs = artist_group[artist_group['Song'].notna() & 
                                 (artist_group['Song'].astype(str).str.strip() != '')]['Song'].unique()
        
        if len(valid_songs) <= 1:
            continue
        
        song_mapping = {}
        processed = set()
        artist_standardizations = 0
        
        for song1 in valid_songs:
            if song1 in processed:
                continue
            
            similar_songs = [song1]
            processed.add(song1)
            
            for song2 in valid_songs:
                if song2 in processed:
                    continue
                
                similarity = fuzz.ratio(str(song1), str(song2))
                if similarity >= threshold:
                    similar_songs.append(song2)
                    processed.add(song2)
            
            if len(similar_songs) > 1:
                # Choose canonical version (most frequently occurring)
                artist_df = result_df[result_df['Artist'] == artist]
                song_counts = artist_df[artist_df['Song'].isin(similar_songs)]['Song'].value_counts()
                canonical = song_counts.index[0]  # Most frequent version
                
                if debug:
                    print(f"🎵 Song standardization for '{artist}': '{canonical}' ← {len(similar_songs)} variations (appears {song_counts.iloc[0]} times)")
                    for song in similar_songs:
                        if song != canonical:
                            print(f"   📝 '{song}' → '{canonical}'")
                
                for song in similar_songs:
                    if song != canonical:
                        song_mapping[song] = canonical
                        artist_standardizations += 1
        
        # Apply song mapping for this artist
        if song_mapping:
            artist_mask = result_df['Artist'] == artist
            result_df.loc[artist_mask, 'Song'] = result_df.loc[artist_mask, 'Song'].map(song_mapping).fillna(result_df.loc[artist_mask, 'Song'])
            total_standardizations += artist_standardizations
    
    print(f"✅ Song standardization: {total_standardizations} changes applied")
    return result_df

def advanced_standardization(df, song_threshold=85, artist_threshold=80, combo_threshold=85, debug=True):
    """
    Advanced standardization that handles songs and artists separately and combined
    
    Args:
        df: DataFrame with Song and Artist columns
        song_threshold: Similarity threshold for song titles
        artist_threshold: Similarity threshold for artist names
        combo_threshold: Similarity threshold for song-artist combinations
        debug: Show detailed information
    
    Returns:
        DataFrame with advanced standardization (same number of rows)
    """
    if not RAPIDFUZZ_AVAILABLE:
        print("⚠️ RapidFuzz not available. Using basic processing.")
        return df
    
    print(f"\n🔧 Advanced fuzzy standardization:")
    print(f"   🎵 Song threshold: {song_threshold}%")
    print(f"   🎤 Artist threshold: {artist_threshold}%")
    print(f"   🎯 Combo threshold: {combo_threshold}%")
    
    original_count = len(df)
    
    # First pass: Standardize artist variations across all songs
    df_artists = standardize_artists(df, artist_threshold, debug)
    
    # Second pass: Standardize song variations within same artist
    df_songs = standardize_songs_by_artist(df_artists, song_threshold, debug)
    
    # Third pass: Final combination standardization
    df_final = standardize_song_artist_combinations(df_songs, combo_threshold, debug=False)
    
    print(f"📊 Advanced standardization complete: {original_count} rows → {len(df_final)} rows")
    print(f"✅ Dataset size unchanged, spellings standardized")
    
    return df_final

def print_standardization_summary(original_df, standardized_df):
    """Print a summary of the standardization results"""
    if original_df.empty or standardized_df.empty:
        return
    
    print(f"\n📊 STANDARDIZATION SUMMARY")
    print(f"=" * 40)
    print(f"📥 Original rows: {len(original_df):,}")
    print(f"📤 Standardized rows: {len(standardized_df):,}")
    print(f"✅ Dataset size: UNCHANGED")
    
    # Count unique combinations before and after
    if 'Song' in original_df.columns and 'Artist' in original_df.columns:
        original_combos = original_df.dropna(subset=['Song', 'Artist'])
        original_unique = len(original_combos.groupby(['Song', 'Artist']))
        
        standardized_combos = standardized_df.dropna(subset=['Song', 'Artist'])
        standardized_unique = len(standardized_combos.groupby(['Song', 'Artist']))
        
        print(f"🎯 Unique song-artist combinations:")
        print(f"   📥 Before: {original_unique:,}")
        print(f"   📤 After: {standardized_unique:,}")
        print(f"   📉 Variations eliminated: {original_unique - standardized_unique:,}")
        print(f"   📈 Standardization efficiency: {((original_unique - standardized_unique) / original_unique * 100):.1f}%")
        
        # Show most frequent songs/artists after standardization
        print(f"\n🎵 TOP SONGS AFTER STANDARDIZATION:")
        top_songs = standardized_df['Song'].value_counts().head(5)
        for song, count in top_songs.items():
            print(f"   {count:3d}x '{song}'")
        
        print(f"\n🎤 TOP ARTISTS AFTER STANDARDIZATION:")
        top_artists = standardized_df['Artist'].value_counts().head(5)
        for artist, count in top_artists.items():
            print(f"   {count:3d}x '{artist}'")

def check_colab_environment():
    """Check if running in Google Colab and setup environment"""
    try:
        # Try importing google.colab - this will only work in Colab
        import google.colab  # type: ignore
        print("🚀 Running in Google Colab")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🔥 GPU available: {gpu_name}")
                print(f"🔥 GPU memory: {gpu_memory:.1f}GB")
                
                # Set GPU memory growth (helps with memory management)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    print("🔥 GPU cache cleared")
            else:
                print("💻 Using CPU (GPU not available)")
        except ImportError:
            print("💻 Using CPU (torch not available)")
            
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"💾 Available memory: {memory.available/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB")
        except ImportError:
            print("💾 Memory info not available")
            
        return True
    except ImportError:
        print("💻 Not running in Colab")
        return False

def mount_drive():
    """Mount Google Drive for file storage"""
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def upload_files():
    """Helper function to upload files in Colab"""
    try:
        from google.colab import files  # type: ignore
        print("📁 Upload your files:")
        uploaded = files.upload()
        return uploaded
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        return {}

def process_single_image_colab(args):
    """Process a single image file - Colab optimized version with GPU support"""
    global ocr_model, DocumentFile
    
    # Ensure DocTR is properly imported
    if ocr_model is None or DocumentFile is None:
        if not setup_doctr():
            print("[Worker] DocTR not available, skipping OCR processing")
            return None
    
    image_path, csv_path, column_boundaries = args
    
    try:
        print(f"[Worker] Processing {os.path.basename(image_path)}")
        
        # Import torch for GPU management
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
        
        # Clear GPU cache before processing (helps with memory management)
        if gpu_available:
            torch.cuda.empty_cache()
        
        # Use the pre-initialized global OCR model (already on GPU if available)
        model = ocr_model
        
        # Run OCR
        doc = DocumentFile.from_images(image_path)
        
        # Process with GPU acceleration if available
        if gpu_available:
            with torch.no_grad():  # Disable gradient computation for inference
                result = model(doc)
        else:
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
        print(f"[Worker] Using OCR version: {OCR_VERSION}")
        df = ocr_boxes_to_dataframe(ocr_results, csv_path, column_boundaries=column_boundaries, image_path=image_path)
        
        # Debug: Check date format in resulting dataframe
        if len(df) > 0 and 'Chart_Date' in df.columns:
            sample_date = df['Chart_Date'].iloc[0] if len(df['Chart_Date']) > 0 else "No date"
            print(f"[Worker] Sample date from processed data: {sample_date}")
        
        # Clean the dataframe
        df = clean_dataframe(df)
        
        # Apply text normalization
        df = normalize_dataframe_text(df)
        
        # Free memory (but don't delete the global model)
        del doc, result, json_output
        
        # Clear GPU cache after processing
        if gpu_available:
            torch.cuda.empty_cache()
        
        print(f"[Worker] Completed {os.path.basename(image_path)}, got {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"[Worker] Error processing {image_path}: {e}")
        # Clear cache on error too
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        return None

def process_date_group_colab(date, filenames, input_dir, max_workers=1):
    """Process a group of images for a specific date - Colab optimized"""
    print(f"\n[COLAB] Processing date: {date} with {len(filenames)} files")
    
    # Use the normalized column boundaries (matching OCR.py)
    column_boundaries = [
        0.021, 0.067, 0.110, 0.434, 0.491, 0.536, 0.572, 0.613, 0.667, 0.706,
        0.767, 0.806, 0.869, 0.908, 0.974
    ]
    
    # Prepare arguments for processing
    process_args = []
    for filename in filenames:
        image_path = os.path.join(input_dir, filename)
        csv_path = f"ocr_table_{filename.replace('.jpg', '.csv')}"
        process_args.append((image_path, csv_path, column_boundaries))
    
    print(f"[COLAB] Using sequential processing for {len(filenames)} files")
    
    # Sequential processing to avoid memory issues
    dataframes = []
    start_time = time.time()
    
    for args in process_args:
        df = process_single_image_colab(args)
        if df is not None:
            dataframes.append(df)
    
    end_time = time.time()
    print(f"[COLAB] Completed {date} in {end_time - start_time:.2f} seconds")
    
    return dataframes

def process_directory_colab(input_dir, output_csv_name, max_workers=1):
    """Process all images in a directory - Colab optimized"""
    
    # Group images by date
    date_groups = {}
    if not os.path.exists(input_dir):
        print(f"[COLAB] Directory {input_dir} not found!")
        return None
        
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        # Extract date from filename (e.g., 2025-08-10_1.jpg -> 2025-08-10)
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})_\d+\.jpg', filename)
        if date_match:
            date = date_match.group(1)
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(filename)

    if not date_groups:
        print(f"[COLAB] No valid images found in {input_dir}")
        return None

    print(f"[COLAB] Found {len(date_groups)} date groups with total {sum(len(files) for files in date_groups.values())} images")
    
    # Process each date group and sort dates in reverse chronological order
    all_date_dataframes = []
    
    # Sort dates first (most recent first)
    sorted_dates = sorted(date_groups.keys(), reverse=True)
    
    for date in sorted_dates:
        filenames = date_groups[date]
        # Sort filenames to ensure consistent order
        filenames.sort()
        
        # Process this date group
        date_dataframes = process_date_group_colab(date, filenames, input_dir, max_workers)
        
        # Apply position-preserving sort for this date
        if date_dataframes:
            merged_date_df = apply_position_preserving_sort(date_dataframes, date)
            all_date_dataframes.append(merged_date_df)

    # Combine all dates into final dataframe (already in chronological order)
    final_df = None
    if all_date_dataframes:
        final_df = pd.concat(all_date_dataframes, ignore_index=True)
        print(f"[COLAB] Final merged dataframe created for {input_dir}")
        print(f"[COLAB] Total rows: {len(final_df)}")
        print(f"[COLAB] Dates processed: {sorted_dates}")
    
    return final_df

def apply_position_preserving_sort(date_dataframes, date):
    """Apply position-preserving sort to dataframes for a single date"""
    merged_date_df = pd.concat(date_dataframes, ignore_index=True)
    
    # Preserve original position for rows with empty ranks
    merged_date_df['original_position'] = merged_date_df.index
    
    # Convert Rank column to numeric for sorting
    def safe_rank_convert(x):
        try:
            if x and str(x).strip() != '':
                return int(x)
            else:
                return None
        except:
            return None
    
    merged_date_df['Rank_numeric'] = merged_date_df['Rank'].apply(safe_rank_convert)
    
    # Separate rows with and without ranks
    rows_with_rank = merged_date_df[merged_date_df['Rank_numeric'].notna()].copy()
    rows_without_rank = merged_date_df[merged_date_df['Rank_numeric'].isna()].copy()
    
    # Sort only rows with valid ranks by rank
    if len(rows_with_rank) > 0:
        rows_with_rank = rows_with_rank.sort_values('Rank_numeric')
    
    # Create final dataframe maintaining original positions for rows without ranks
    final_rows = []
    
    # Create a mapping of original positions for rows without ranks
    no_rank_positions = {}
    for idx, row in rows_without_rank.iterrows():
        original_pos = row['original_position']
        no_rank_positions[original_pos] = row
    
    # Create a list of ranked rows to insert
    ranked_rows_list = list(rows_with_rank.iterrows()) if len(rows_with_rank) > 0 else []
    ranked_idx = 0
    
    # Go through all original positions
    for pos in range(len(merged_date_df)):
        if pos in no_rank_positions:
            # Keep row without rank in original position
            final_rows.append(no_rank_positions[pos])
        else:
            # Insert next ranked row
            if ranked_idx < len(ranked_rows_list):
                final_rows.append(ranked_rows_list[ranked_idx][1])
                ranked_idx += 1
    
    # Convert back to DataFrame
    if final_rows:
        result_df = pd.DataFrame(final_rows)
        
        # Drop helper columns
        columns_to_drop = ['Rank_numeric', 'original_position']
        columns_to_drop = [col for col in columns_to_drop if col in result_df.columns]
        if columns_to_drop:
            result_df = result_df.drop(columns_to_drop, axis=1)
        
        result_df = result_df.reset_index(drop=True)
        
        # Debug output to verify sorting
        ranks_sample = result_df['Rank'].head(10).tolist()
        print(f"[COLAB] Merged dataframes for {date}")
        print(f"[COLAB] First 10 ranks after sorting: {ranks_sample}")
        
        return result_df
    
    return merged_date_df

def merge_all_csvs_in_chart_dfs_colab(chart_dfs_dir):
    """Merge all CSV files in chart_dfs directory in correct order - Colab version"""
    print(f"\n[MERGE] Starting merge of all CSV files in {chart_dfs_dir}")
    
    if not os.path.exists(chart_dfs_dir):
        print(f"[MERGE] Directory {chart_dfs_dir} not found!")
        return None
    
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
        # Try different date formats - updated for mm/dd/yy format
        combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], format='%m/%d/%y', errors='coerce')
        if combined_df['Chart_Date_parsed'].isna().all():
            combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], format='%d-%m-%y', errors='coerce')
        if combined_df['Chart_Date_parsed'].isna().all():
            combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], format='%Y-%m-%d', errors='coerce')
        if combined_df['Chart_Date_parsed'].isna().all():
            combined_df['Chart_Date_parsed'] = pd.to_datetime(combined_df['Chart_Date'], errors='coerce')
    except Exception as e:
        print(f"[MERGE] Warning: Could not parse dates properly: {e}")
        # Fallback: sort by Chart_Date as string
        combined_df['Chart_Date_parsed'] = combined_df['Chart_Date']
    
    # Convert Rank to numeric for sorting, but preserve original positions for empty ranks
    def safe_rank_convert(x):
        try:
            if x and str(x).strip() != '':
                return int(x)
            else:
                return None  # Keep as None instead of inf to preserve position
        except:
            return None
    
    combined_df['Rank_numeric'] = combined_df['Rank'].apply(safe_rank_convert)
    
    # Sort dates first, then apply position-preserving sort within each date
    unique_dates = combined_df['Chart_Date_parsed'].dropna().unique()
    sorted_dates = sorted(unique_dates, reverse=True)  # Most recent first
    
    print(f"[MERGE] Processing {len(sorted_dates)} dates in chronological order")
    
    final_dfs = []
    
    for date in sorted_dates:
        print(f"[MERGE] Processing date: {date}")
        
        date_df = combined_df[combined_df['Chart_Date_parsed'] == date].copy()
        if len(date_df) == 0:
            continue
        
        # Apply position-preserving sort for this date
        sorted_date_df = apply_position_preserving_sort_merge(date_df)
        final_dfs.append(sorted_date_df)
        print(f"[MERGE] Completed date {date}: {len(sorted_date_df)} rows")
    
    # Combine all dates in chronological order
    if final_dfs:
        final_df = pd.concat(final_dfs, ignore_index=True)
        print(f"[MERGE] Successfully combined {len(final_dfs)} dates in chronological order")
    else:
        print(f"[MERGE] No valid data found, using original combined dataframe")
        final_df = combined_df
    
    # Drop helper columns
    helper_columns = ['Chart_Date_parsed', 'Rank_numeric']
    columns_to_drop = [col for col in helper_columns if col in final_df.columns]
    if columns_to_drop:
        final_df = final_df.drop(columns_to_drop, axis=1).reset_index(drop=True)
    
    print(f"[MERGE] Final merged dataframe: {len(final_df)} rows")
    if not final_df.empty:
        print(f"[MERGE] Date range: {final_df['Chart_Date'].min()} to {final_df['Chart_Date'].max()}")
        
        # Debug: Show sample of ranks for verification
        sample_ranks = final_df['Rank'].head(20).tolist()
        print(f"[MERGE] First 20 ranks after merge: {sample_ranks}")
        
        # Debug: Show date sequence verification
        unique_dates_in_final = final_df['Chart_Date'].unique()[:5]  # First 5 dates
        print(f"[MERGE] First 5 dates in final sequence: {list(unique_dates_in_final)}")
    
    return final_df

def apply_position_preserving_sort_merge(date_df):
    """Apply position-preserving sort for merge function"""
    date_df = date_df.reset_index(drop=True)
    date_df['date_position'] = range(len(date_df))
    
    # Separate rows with and without ranks
    rows_with_rank = date_df[date_df['Rank_numeric'].notna()].copy()
    rows_without_rank = date_df[date_df['Rank_numeric'].isna()].copy()
    
    # Sort only rows with valid ranks by rank
    if len(rows_with_rank) > 0:
        rows_with_rank = rows_with_rank.sort_values('Rank_numeric')
    
    # Create final dataframe maintaining original positions for rows without ranks
    final_rows = []
    
    # Create a mapping of original positions for rows without ranks
    no_rank_positions = {}
    for idx, row in rows_without_rank.iterrows():
        original_pos = row['date_position']
        no_rank_positions[original_pos] = row
    
    # Create a list of ranked rows to insert
    ranked_rows_list = list(rows_with_rank.iterrows()) if len(rows_with_rank) > 0 else []
    ranked_idx = 0
    
    # Go through all original positions within this date
    for pos in range(len(date_df)):
        if pos in no_rank_positions:
            # Keep row without rank in original position
            final_rows.append(no_rank_positions[pos])
        else:
            # Insert next ranked row
            if ranked_idx < len(ranked_rows_list):
                final_rows.append(ranked_rows_list[ranked_idx][1])
                ranked_idx += 1
    
    # Convert back to DataFrame for this date
    if final_rows:
        result_df = pd.DataFrame(final_rows)
        # Drop helper columns for this date
        if 'date_position' in result_df.columns:
            result_df = result_df.drop('date_position', axis=1)
        return result_df
    
    return date_df

def save_to_drive(local_path, drive_path):
    """Save file to Google Drive"""
    try:
        import shutil
        drive_full_path = f"/content/drive/MyDrive/{drive_path}"
        os.makedirs(os.path.dirname(drive_full_path), exist_ok=True)
        shutil.copy2(local_path, drive_full_path)
        print(f"✅ Saved to Google Drive: {drive_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving to Drive: {e}")
        return False

def download_file(file_path):
    """Download file in Colab"""
    try:
        from google.colab import files  # type: ignore
        files.download(file_path)
        print(f"✅ Downloaded: {file_path}")
    except Exception as e:
        print(f"❌ Error downloading file: {e}")

def verify_and_convert_date_format(df):
    """Verify date format and convert from dd-mm-yy to mm/dd/yy if needed"""
    if 'Chart_Date' not in df.columns:
        print("⚠️ No Chart_Date column found, skipping date format verification")
        return df
    
    print("\n📅 Verifying date format...")
    
    # Get sample of dates to check format
    sample_dates = df['Chart_Date'].dropna().head(5).tolist()
    print(f"📊 Sample dates: {sample_dates}")
    
    # Check if dates are in dd-mm-yy format (need conversion)
    dd_mm_yy_pattern = r'^\d{1,2}-\d{1,2}-\d{2}$'
    mm_dd_yy_pattern = r'^\d{1,2}/\d{1,2}/\d{2}$'
    
    needs_conversion = False
    if sample_dates:
        first_date = str(sample_dates[0])
        if re.match(dd_mm_yy_pattern, first_date):
            needs_conversion = True
            print(f"🔄 Detected dd-mm-yy format, conversion needed")
        elif re.match(mm_dd_yy_pattern, first_date):
            print(f"✅ Already in mm/dd/yy format")
        else:
            print(f"⚠️ Unknown date format: {first_date}")
    
    if needs_conversion:
        print("🔄 Converting dates from dd-mm-yy to mm/dd/yy format...")
        
        # Count how many dates will be converted
        total_dates = len(df['Chart_Date'].dropna())
        
        def convert_date(date_str):
            if pd.isna(date_str) or date_str == '':
                return date_str
            
            # Match dd-mm-yy pattern
            match = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{2})$', str(date_str))
            if match:
                day, month, year = match.groups()
                return f"{month}/{day}/{year}"  # Convert to mm/dd/yy
            else:
                return date_str  # Return unchanged if no match
        
        # Apply conversion
        df['Chart_Date'] = df['Chart_Date'].apply(convert_date)
        
        # Show sample of converted dates
        converted_sample = df['Chart_Date'].dropna().head(5).tolist()
        print(f"✅ Converted {total_dates} dates from dd-mm-yy to mm/dd/yy")
        print(f"📊 Sample converted dates: {converted_sample}")
        print(f"📅 Date format conversion completed: dd-mm-yy → mm/dd/yy")
    
    return df

# Main execution function for Colab
def run_ocr_batch_processing_colab():
    """Main function to run OCR batch processing in Google Colab"""
    
    print("🚀 OCR Batch Processing for Google Colab")
    print("=" * 50)
    
    # First, setup DocTR
    print("\n🔧 Setting up DocTR...")
    if not setup_doctr():
        print("❌ Failed to setup DocTR. Please install it first:")
        print("!pip install doctr[torch]")
        return None
    
    # Check environment
    is_colab = check_colab_environment()
    
    # Setup directories
    base_dir = "."
    sorted_images_dir = f"{base_dir}/sorted_images"
    chart_dfs_dir = f"{base_dir}/chart_dfs"
    
    # Create directories
    os.makedirs(chart_dfs_dir, exist_ok=True)
    
    print(f"\n📁 Working directories:")
    print(f"   - Base: {base_dir}")
    print(f"   - Images: {sorted_images_dir}")
    print(f"   - Output: {chart_dfs_dir}")
    print(f"   - DocTR initialized: {ocr_model is not None}")
    print(f"   - Environment: {'Google Colab' if is_colab else 'Local/Other'}")
    print(f"   - Row tolerance: {DYNAMIC_ROW_TOLERANCE_PIXELS} pixels")
    
    # Colab-specific settings - always use sequential processing
    max_workers = 1  # Single worker to avoid memory issues
    
    # Check if images directory exists
    if not os.path.exists(sorted_images_dir):
        print(f"\n⚠️ {sorted_images_dir} not found.")
        print("Please upload your images directory or check available directories:")
        
        # Show available directories
        available_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Available directories: {available_dirs}")
        
        # Try alternative directory
        alt_dir = f"{base_dir}/charts_images"
        if os.path.exists(alt_dir):
            sorted_images_dir = alt_dir
            print(f"✅ Using alternative directory: {alt_dir}")
        else:
            print("❌ No image directories found. Please upload your images.")
            return None
    
    # Check for subdirectories vs single directory
    subdirs = [d for d in os.listdir(sorted_images_dir) 
              if os.path.isdir(os.path.join(sorted_images_dir, d))]
    
    if subdirs:
        print(f"\n📂 Found {len(subdirs)} subdirectories: {subdirs}")
        
        # Process each subdirectory
        for subdir in subdirs:
            print(f"\n{'='*30}")
            print(f"Processing subdirectory: {subdir}")
            print(f"{'='*30}")
            
            subdir_path = os.path.join(sorted_images_dir, subdir)
            output_csv_name = f"{subdir.replace(' ', '_')}_merged.csv"
            
            # Process the subdirectory
            start_time = time.time()
            final_df = process_directory_colab(subdir_path, output_csv_name, max_workers)
            end_time = time.time()
            
            if final_df is not None:
                # Save to chart_dfs directory
                chart_dfs_csv_path = os.path.join(chart_dfs_dir, output_csv_name)
                final_df.to_csv(chart_dfs_csv_path, index=False)
                print(f"✅ CSV saved: {chart_dfs_csv_path}")
                print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
            else:
                print(f"❌ No data processed for subdirectory: {subdir}")
    else:
        # Process single directory
        print(f"\n📂 Processing single directory: {sorted_images_dir}")
        
        start_time = time.time()
        final_df = process_directory_colab(sorted_images_dir, "charts_images_merged.csv", max_workers)
        end_time = time.time()
        
        if final_df is not None:
            chart_dfs_path = os.path.join(chart_dfs_dir, "charts_images_merged.csv")
            final_df.to_csv(chart_dfs_path, index=False)
            print(f"✅ CSV saved: {chart_dfs_path}")
            print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
    
    # Final merge step
    print(f"\n🔄 Starting final merge...")
    final_merge_start = time.time()
    
    final_complete_df = merge_all_csvs_in_chart_dfs_colab(chart_dfs_dir)
    
    if final_complete_df is not None:
        # Verify and convert date format before saving final files
        final_complete_df = verify_and_convert_date_format(final_complete_df)
        
        # Apply final text normalization to ensure consistency
        final_complete_df = normalize_dataframe_text(final_complete_df, enable_fuzzy_standardization=False)
        
        # Apply fuzzy standardization to the final merged dataset
        if RAPIDFUZZ_AVAILABLE:
            print(f"\n🔧 Applying fuzzy standardization to final dataset...")
            original_df_for_comparison = final_complete_df.copy()
            final_complete_df = advanced_standardization(
                final_complete_df, 
                song_threshold=85, 
                artist_threshold=80, 
                combo_threshold=85, 
                debug=True
            )
            print_standardization_summary(original_df_for_comparison, final_complete_df)
        else:
            print("⚠️ RapidFuzz not available. Skipping fuzzy standardization.")
            print("Install with: !pip install rapidfuzz")
        
        # Save final merged files in both CSV and XLSX formats
        final_csv_path = f"{base_dir}/final_merged_all_dates.csv"
        final_xlsx_path = f"{base_dir}/final_merged_all_dates.xlsx"
        
        # Save CSV
        final_complete_df.to_csv(final_csv_path, index=False)
        print(f"✅ Final CSV saved: {final_csv_path}")
        
        # Save XLSX
        try:
            final_complete_df.to_excel(final_xlsx_path, index=False, engine='openpyxl')
            print(f"✅ Final XLSX saved: {final_xlsx_path}")
        except ImportError:
            print("⚠️ openpyxl not installed. Installing...")
            import subprocess
            subprocess.run(['pip', 'install', 'openpyxl'], check=True)
            final_complete_df.to_excel(final_xlsx_path, index=False, engine='openpyxl')
            print(f"✅ Final XLSX saved: {final_xlsx_path}")
        except Exception as e:
            print(f"❌ Error saving XLSX: {e}")
        
        # Save to chart_dfs directory
        chart_dfs_final_csv = os.path.join(chart_dfs_dir, "final_merged_all_dates.csv")
        chart_dfs_final_xlsx = os.path.join(chart_dfs_dir, "final_merged_all_dates.xlsx")
        
        final_complete_df.to_csv(chart_dfs_final_csv, index=False)
        try:
            final_complete_df.to_excel(chart_dfs_final_xlsx, index=False, engine='openpyxl')
        except:
            pass  # Already handled above
        
        final_merge_end = time.time()
        print(f"⏱️ Final merge time: {final_merge_end - final_merge_start:.2f} seconds")
        print(f"📊 Total rows: {len(final_complete_df)}")
        
        # Show sample of final data
        if len(final_complete_df) > 0:
            print(f"\n📋 Sample of final merged data:")
            sample_cols = ['Chart_Date', 'Rank', 'Song', 'Artist']
            available_cols = [col for col in sample_cols if col in final_complete_df.columns]
            if available_cols:
                print(final_complete_df[available_cols].head(10).to_string(index=False))
        
        # Save to Google Drive if mounted
        if os.path.exists("/content/drive"):
            save_to_drive(final_csv_path, "twitter_scraper/final_merged_all_dates.csv")
            save_to_drive(final_xlsx_path, "twitter_scraper/final_merged_all_dates.xlsx")
        
        # Offer download for both files
        if is_colab:
            print(f"\n⬇️ Downloading final files...")
            download_file(final_csv_path)
            download_file(final_xlsx_path)
            
        return final_complete_df
    else:
        print(f"❌ No data available for final merge")
        return None

def run_musicbrainz_canonicalization_colab(input_csv_path="final_merged_all_dates.csv", 
                                           output_csv_path=None,
                                           confidence_threshold=75,  # Lowered from 85 for better match rate
                                           individual_score_threshold=75,  # NEW: Minimum score for both song and artist
                                           enable_pre_filter=True,
                                           resume_from_checkpoint=True):
    """
    Run MusicBrainz canonicalization in Google Colab - designed for separate cell execution
    
    Args:
        input_csv_path: Path to input CSV file (default: "final_merged_all_dates.csv")
        output_csv_path: Path to output CSV file (optional, auto-generated if None)
        confidence_threshold: Minimum confidence score for matches (80-95, default 75)
        individual_score_threshold: Minimum score required for both song and artist (default 75)
        enable_pre_filter: Whether to pre-filter non-music entries (default True)
        resume_from_checkpoint: Whether to resume from existing checkpoint (default True)
    
    Returns:
        Path to canonical CSV file or None if failed
    """
    
    print("🎵 MusicBrainz Canonicalization for Google Colab")
    print("=" * 55)
    
    # Check if running in Colab
    is_colab = check_colab_environment()
    
    # Install required packages if needed
    print("\n📦 Checking dependencies...")
    try:
        import requests
        print("✅ requests available")
    except ImportError:
        print("⚠️ Installing requests...")
        import subprocess
        subprocess.run(['pip', 'install', 'requests'], check=True)
        import requests
        print("✅ requests installed")
    
    # Check if input file exists
    if not os.path.exists(input_csv_path):
        print(f"❌ Input file not found: {input_csv_path}")
        print("Available files in current directory:")
        for file in os.listdir("."):
            if file.endswith('.csv'):
                print(f"   📄 {file}")
        return None
    
    # Auto-generate output path if not provided
    if output_csv_path is None:
        base_name = os.path.splitext(input_csv_path)[0]
        output_csv_path = f"{base_name}_canonical.csv"
    
    print(f"\n📁 Input file: {input_csv_path}")
    print(f"📁 Output file: {output_csv_path}")
    print(f"🎯 Confidence threshold: {confidence_threshold}%")
    print(f"🎵 Individual score threshold: {individual_score_threshold}% (both song and artist must meet this)")
    print(f"🔍 Pre-filtering: {'Enabled' if enable_pre_filter else 'Disabled'}")
    print(f"💾 Resume from checkpoint: {'Yes' if resume_from_checkpoint else 'No'}")
    
    try:
        # Try to create the canonicalization class inline for Colab
        print("\n🔧 Setting up MusicBrainz canonicalizer...")
        
        # Create a simplified version for Colab
        canonical_df = run_simple_musicbrainz_canonicalization(
            input_csv_path=input_csv_path,
            output_csv_path=output_csv_path,
            confidence_threshold=confidence_threshold,
            individual_score_threshold=individual_score_threshold,  # Pass the parameter
            enable_pre_filter=enable_pre_filter,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        if canonical_df is not None:
            # Save Excel version
            xlsx_path = output_csv_path.replace('.csv', '.xlsx')
            try:
                canonical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
                print(f"✅ Excel version saved: {xlsx_path}")
            except Exception as e:
                print(f"⚠️ Could not save Excel version: {e}")
            
            # Save to Google Drive if mounted
            if os.path.exists("/content/drive"):
                print("\n☁️ Saving to Google Drive...")
                save_to_drive(output_csv_path, f"twitter_scraper/{os.path.basename(output_csv_path)}")
                if os.path.exists(xlsx_path):
                    save_to_drive(xlsx_path, f"twitter_scraper/{os.path.basename(xlsx_path)}")
            
            # Download files in Colab
            if is_colab:
                print(f"\n⬇️ Downloading canonical files...")
                download_file(output_csv_path)
                if os.path.exists(xlsx_path):
                    download_file(xlsx_path)
            
            print(f"\n🎉 MusicBrainz canonicalization completed successfully!")
            print(f"📊 Canonical dataset: {len(canonical_df)} rows")
            return output_csv_path
        else:
            print(f"❌ Canonicalization failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during canonicalization: {e}")
        print(f"💡 Tip: Make sure you have a stable internet connection")
        print(f"💡 Tip: Try reducing confidence_threshold or disabling pre_filter")
        return None


def run_simple_musicbrainz_canonicalization(input_csv_path, output_csv_path, confidence_threshold=75,  # Lowered from 85
                                           individual_score_threshold=75,  # Added parameter
                                           enable_pre_filter=True, resume_from_checkpoint=True):
    """
    Simplified MusicBrainz canonicalization for Google Colab
    This is a lightweight version that doesn't require external files
    
    Args:
        individual_score_threshold: Minimum score required for both song and artist components
    """
    import requests
    import time
    import json
    from datetime import datetime
    import pickle
    from pathlib import Path
    import logging
    
    # Setup logging to file
    log_file_path = output_csv_path.replace('.csv', '_canonicalization.log')
    
    # Create a logger
    logger = logging.getLogger('musicbrainz_canonicalization')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(logging.Formatter('%(message)s'))  # Simpler console format
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    def log_print(message):
        """Helper function to log to both file and console"""
        logger.info(message)
    
    log_print("🎵 Starting MusicBrainz Canonicalization")
    log_print(f"📁 Input: {input_csv_path}")
    log_print(f"📁 Output: {output_csv_path}")
    log_print(f"📁 Log file: {log_file_path}")
    log_print(f"🎯 Confidence threshold: {confidence_threshold}%")
    log_print(f"🎵 Individual score threshold: {individual_score_threshold}%")
    
    # Configuration
    MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
    USER_AGENT = "ChartDataProcessor/1.0 (https://github.com/yourproject/charts)"
    RATE_LIMIT_DELAY = 1.1
    MAX_RETRIES = 3
    CHECKPOINT_INTERVAL = 25  # More frequent checkpoints for Colab
    
    # Setup checkpoint directory
    checkpoint_dir = Path("mb_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize caches
    api_cache = {}  # Stores raw MusicBrainz API responses (threshold-independent)
    canonical_cache = {}  # Stores processed canonical results (threshold-specific)
    failed_queries = set()  # Stores failed API queries
    
    # Load existing caches
    cache_files = {
        'api_cache.pkl': api_cache,
        'canonical_cache.pkl': canonical_cache,
        'failed_queries.pkl': failed_queries
    }
    
    for filename, cache_dict in cache_files.items():
        filepath = checkpoint_dir / filename
        if filepath.exists() and resume_from_checkpoint:
            try:
                with open(filepath, 'rb') as f:
                    loaded_data = pickle.load(f)
                    cache_dict.update(loaded_data)
                log_print(f"✅ Loaded cache {filename}: {len(loaded_data)} entries")
            except Exception as e:
                log_print(f"⚠️ Error loading {filename}: {e}")
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'User-Agent': USER_AGENT,
        'Accept': 'application/json'
    })
    
    def get_combo_hash(song, artist, threshold=None):
        import hashlib
        combo_str = f"{str(song).lower().strip()}|{str(artist).lower().strip()}"
        if threshold is not None:
            combo_str += f"|{threshold}"  # Include threshold in hash for cache differentiation
        return hashlib.md5(combo_str.encode()).hexdigest()
    
    def normalize_for_search(text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).strip()
        text = re.sub(r'\b(feat\.?|featuring|ft\.?)\s+.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
        text = re.sub(r'\s*\[[^\]]*\]\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        special_chars = r'[+\-&|!(){}[\]^"~*?:\\]'
        text = re.sub(special_chars, r'\\\g<0>', text)
        return text
    
    def quick_confidence_check(data, song, artist, min_threshold=35):
        """Quick confidence check to see if results are worth keeping or if we should try next strategy"""
        if not data or 'recordings' not in data or not data['recordings']:
            return 0
        
        # Check the best result only for speed
        recording = data['recordings'][0]
        try:
            mb_song = recording.get('title', '')
            mb_artist = ''
            
            if 'artist-credit' in recording and recording['artist-credit']:
                artists = []
                for credit in recording['artist-credit']:
                    if isinstance(credit, dict) and 'artist' in credit:
                        artists.append(credit['artist'].get('name', ''))
                    elif isinstance(credit, str):
                        artists.append(credit)
                mb_artist = ', '.join(filter(None, artists))
            
            if not mb_song or not mb_artist:
                return 0
            
            # Quick similarity calculation
            if RAPIDFUZZ_AVAILABLE:
                song_score = calculate_rapidfuzz_similarity(song, mb_song)
                artist_score = calculate_rapidfuzz_similarity(artist, mb_artist)
            else:
                song_score = calculate_enhanced_similarity(song, mb_song)
                artist_score = calculate_enhanced_similarity(artist, mb_artist)
            
            # Return the average confidence
            return (song_score + artist_score) / 2
            
        except Exception:
            return 0
    
    def query_musicbrainz(song, artist):
        combo_hash = get_combo_hash(song, artist)  # API cache doesn't need threshold
        
        if combo_hash in failed_queries:
            return None
        if combo_hash in api_cache:
            return api_cache[combo_hash]
        
        norm_song = normalize_for_search(song)
        norm_artist = normalize_for_search(artist)
        
        if not norm_song or not norm_artist:
            failed_queries.add(combo_hash)
            return None
        
        # Try multiple search strategies in order of preference
        search_strategies = [
            # Strategy 1: Partial matching without quotes for OCR errors (DEFAULT - most flexible)
            f'recording:{norm_song} AND artist:{norm_artist}',
            
            # Strategy 2: Exact match (most precise, but often fails with OCR data)
            f'recording:"{norm_song}" AND artist:"{norm_artist}"',
            
            # Strategy 3: OR search with both terms (wider net)
            f'recording:"{norm_song}" OR artist:"{norm_artist}"',
            
            # Strategy 4: Separate searches combined (very wide)
            f'recording:{norm_song} OR artist:{norm_artist}',
            
            # Strategy 5: Just song title search with quotes
            f'recording:"{norm_song}"',
            
            # Strategy 6: Just artist search with quotes
            f'artist:"{norm_artist}"',
            
            # Strategy 7: Partial song search without quotes
            f'recording:{norm_song}',
            
            # Strategy 8: Partial artist search without quotes
            f'artist:{norm_artist}'
        ]
        
        # Increase result limit significantly
        params = {'limit': 50, 'fmt': 'json'}  # Increased from 25 to 50 for more thorough matching
        
        for strategy_idx, query in enumerate(search_strategies):
            params['query'] = query
            
            for attempt in range(MAX_RETRIES):
                try:
                    response = session.get(f"{MUSICBRAINZ_BASE_URL}/recording", params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Check if we got meaningful results
                        if data.get('recordings') and len(data['recordings']) > 0:
                            # For the first strategy (partial matching), accept any results
                            # For other strategies, check confidence
                            if strategy_idx == 0:
                                # Always accept partial matching results (strategy 1)
                                api_cache[combo_hash] = data
                                time.sleep(RATE_LIMIT_DELAY)
                                return data
                            else:
                                # Quick confidence check for other strategies
                                confidence = quick_confidence_check(data, norm_song, norm_artist, min_threshold=35)
                                
                                if confidence >= 35 or strategy_idx >= len(search_strategies) - 1:
                                    # Good confidence or last strategy - cache and return
                                    api_cache[combo_hash] = data
                                    time.sleep(RATE_LIMIT_DELAY)
                                    return data
                                elif strategy_idx < len(search_strategies) - 1:
                                    # Low confidence - try next strategy
                                    log_print(f"🔄 Strategy {strategy_idx + 1} confidence too low ({confidence:.1f}%), trying next strategy")
                                    break
                        elif strategy_idx < len(search_strategies) - 1:
                            # Try next strategy if this one returned empty
                            break
                            
                    elif response.status_code == 429:
                        wait_time = 5 * (attempt + 1)
                        print(f"⏳ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        if attempt == MAX_RETRIES - 1:
                            break  # Try next strategy
                        time.sleep(RATE_LIMIT_DELAY)
                        
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        break  # Try next strategy
                    time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
            
            time.sleep(RATE_LIMIT_DELAY)  # Delay between strategies
        
        # If all strategies failed, mark as failed
        failed_queries.add(combo_hash)  # Use API cache hash (without threshold)
        return None
    
    def calculate_enhanced_similarity(s1, s2):
        """Enhanced similarity calculation using Levenshtein distance and other approaches"""
        s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
        
        # Exact match
        if s1 == s2:
            return 100
        
        # Remove common variations for comparison
        def normalize_for_comparison(text):
            text = re.sub(r'\b(feat\.?|featuring|ft\.?)\s+.*$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*\([^)]*\)', '', text)  # Remove parentheses
            text = re.sub(r'\s*\[[^\]]*\]', '', text)  # Remove brackets
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        norm_s1 = normalize_for_comparison(s1)
        norm_s2 = normalize_for_comparison(s2)
        
        # Check normalized exact match
        if norm_s1 == norm_s2:
            return 95
        
        # Levenshtein distance-based similarity (most accurate)
        def levenshtein_similarity(a, b):
            if not a or not b:
                return 0
            if a == b:
                return 100
            
            # Calculate Levenshtein distance
            max_len = max(len(a), len(b))
            if max_len == 0:
                return 100
            
            # Simple Levenshtein distance calculation
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(a, b)
            similarity = ((max_len - distance) / max_len) * 100
            return max(0, similarity)
        
        # Jaccard similarity for word-based comparison
        def jaccard_similarity(a, b):
            words_a = set(a.split())
            words_b = set(b.split())
            
            if not words_a or not words_b:
                return 0
            
            intersection = len(words_a.intersection(words_b))
            union = len(words_a.union(words_b))
            
            return (intersection / union * 100) if union > 0 else 0
        
        # Subsequence similarity
        def longest_common_subsequence_similarity(a, b):
            def lcs_length(s1, s2):
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            if not a or not b:
                return 0
            
            lcs_len = lcs_length(a, b)
            max_len = max(len(a), len(b))
            return (lcs_len / max_len * 100) if max_len > 0 else 0
        
        # Check if one string contains the other (high confidence for substrings)
        def containment_score(a, b):
            if a in b or b in a:
                shorter = min(len(a), len(b))
                longer = max(len(a), len(b))
                return (shorter / longer * 100) if longer > 0 else 0
            return 0
        
        # Calculate all similarity metrics
        levenshtein_score = levenshtein_similarity(norm_s1, norm_s2)
        jaccard_score = jaccard_similarity(norm_s1, norm_s2)
        lcs_score = longest_common_subsequence_similarity(norm_s1, norm_s2)
        contain_score = containment_score(norm_s1, norm_s2)
        
        # Use weighted combination with Levenshtein as primary metric
        final_score = (
            levenshtein_score * 0.4 +    # Primary: character-level similarity
            jaccard_score * 0.25 +       # Secondary: word-level similarity
            lcs_score * 0.2 +             # Tertiary: sequence similarity
            contain_score * 0.15          # Quaternary: containment bonus
        )
        
        return min(100, max(0, final_score))
    
    def calculate_rapidfuzz_similarity(s1, s2):
        """Use RapidFuzz for more sophisticated similarity calculation"""
        if not RAPIDFUZZ_AVAILABLE:
            return calculate_enhanced_similarity(s1, s2)
        
        s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
        
        # Exact match
        if s1 == s2:
            return 100
        
        # Normalize for comparison
        def normalize_for_comparison(text):
            text = re.sub(r'\b(feat\.?|featuring|ft\.?)\s+.*$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*\([^)]*\)', '', text)
            text = re.sub(r'\s*\[[^\]]*\]', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        norm_s1 = normalize_for_comparison(s1)
        norm_s2 = normalize_for_comparison(s2)
        
        if norm_s1 == norm_s2:
            return 95
        
        # Use multiple RapidFuzz algorithms
        try:
            from rapidfuzz import fuzz
            
            # Ratio: Overall similarity
            ratio_score = fuzz.ratio(norm_s1, norm_s2)
            
            # Partial ratio: Best matching substring
            partial_score = fuzz.partial_ratio(norm_s1, norm_s2)
            
            # Token sort: Ignore word order
            token_sort_score = fuzz.token_sort_ratio(norm_s1, norm_s2)
            
            # Token set: Ignore duplicates and order
            token_set_score = fuzz.token_set_ratio(norm_s1, norm_s2)
            
            # WRatio: Weighted combination of all methods
            wratio_score = fuzz.WRatio(norm_s1, norm_s2)
            
            # Use weighted combination with WRatio as primary
            final_score = (
                wratio_score * 0.4 +       # Primary: RapidFuzz's best algorithm
                ratio_score * 0.25 +       # Secondary: Overall similarity
                token_sort_score * 0.2 +   # Tertiary: Order-independent
                partial_score * 0.1 +      # Quaternary: Substring match
                token_set_score * 0.05     # Quinary: Set-based match
            )
            
            return min(100, max(0, final_score))
            
        except Exception:
            # Fallback to enhanced similarity if RapidFuzz fails
            return calculate_enhanced_similarity(s1, s2)
    
    def extract_canonical_names(mb_data, original_song, original_artist, individual_score_threshold=75):
        if not mb_data or 'recordings' not in mb_data:
            return None, None  # Return tuple: (accepted_match, best_match)
        
        recordings = mb_data['recordings']
        if not recordings:
            return None, None
        
        best_match = None
        best_score = 0
        
        # Process more results for thorough matching
        for recording in recordings[:25]:  # Increased from 15 to 25 for better coverage
            try:
                mb_song = recording.get('title', '')
                mb_artist = ''
                
                if 'artist-credit' in recording and recording['artist-credit']:
                    artists = []
                    for credit in recording['artist-credit']:
                        if isinstance(credit, dict) and 'artist' in credit:
                            artists.append(credit['artist'].get('name', ''))
                        elif isinstance(credit, str):
                            artists.append(credit)
                    mb_artist = ', '.join(filter(None, artists))
                
                if not mb_song or not mb_artist:
                    continue
                
                # Use RapidFuzz similarity if available, otherwise enhanced similarity
                if RAPIDFUZZ_AVAILABLE:
                    song_score = calculate_rapidfuzz_similarity(original_song, mb_song)
                    artist_score = calculate_rapidfuzz_similarity(original_artist, mb_artist)
                else:
                    song_score = calculate_enhanced_similarity(original_song, mb_song)
                    artist_score = calculate_enhanced_similarity(original_artist, mb_artist)
                
                # Enhanced weighted scoring with bonus for exact matches
                if song_score == 100 and artist_score >= 85:
                    combined_score = 98  # Very high score for exact song match
                elif artist_score == 100 and song_score >= 85:
                    combined_score = 95  # High score for exact artist match
                elif song_score >= 95 and artist_score >= 95:
                    combined_score = 96  # High score for near-exact matches
                else:
                    # Weighted scoring: song is more important than artist for identification
                    combined_score = (song_score * 0.65 + artist_score * 0.35)
                
                # Track the best match regardless of thresholds
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        'song': mb_song,
                        'artist': mb_artist,
                        'score': combined_score,
                        'mbid': recording.get('id', ''),
                        'song_score': song_score,
                        'artist_score': artist_score,
                        'algorithm': 'RapidFuzz' if RAPIDFUZZ_AVAILABLE else 'Enhanced'
                    }
            except Exception:
                continue
        
        # Now check if the best match meets the acceptance criteria
        if best_match:
            threshold_with_tolerance = max(50, confidence_threshold - 15)  # More lenient threshold
            
            # REQUIREMENT: Both song and artist scores must meet individual threshold
            meets_individual_thresholds = (best_match['song_score'] >= individual_score_threshold and 
                                          best_match['artist_score'] >= individual_score_threshold)
            
            meets_combined_threshold = best_match['score'] >= threshold_with_tolerance
            
            if meets_individual_thresholds and meets_combined_threshold:
                return best_match, best_match  # Both accepted and best are the same
            else:
                return None, best_match  # Rejected but we still have the best match
        
        return None, None
    
    def get_canonical_names(song, artist, threshold=None):
        # Use threshold-aware hash for canonical cache (allows reuse of API data with different thresholds)
        combo_hash_with_threshold = get_combo_hash(song, artist, threshold)
        
        if combo_hash_with_threshold in canonical_cache:
            cached_result = canonical_cache[combo_hash_with_threshold]
            if cached_result:
                algorithm = cached_result.get('algorithm', 'Unknown')
                log_print(f"💾 Cached ({algorithm}): '{song}' → '{cached_result['song']}' | '{artist}' → '{cached_result['artist']}' (score: {cached_result['score']:.1f})")
            return cached_result
        
        # Query API using threshold-independent hash (reuse API data)
        mb_data = query_musicbrainz(song, artist)
        if not mb_data:
            canonical_cache[combo_hash_with_threshold] = None
            log_print(f"❌ No results: '{song}' by '{artist}'")
            return None
        
        # Apply threshold to cached API data - get both accepted and best match
        canonical, best_match = extract_canonical_names(mb_data, song, artist, threshold or individual_score_threshold)
        canonical_cache[combo_hash_with_threshold] = canonical
        
        actual_threshold = threshold or individual_score_threshold
        
        if canonical:
            # Accepted match
            algorithm = canonical.get('algorithm', 'Unknown')
            log_print(f"✅ ({algorithm}) '{song}' → '{canonical['song']}' | '{artist}' → '{canonical['artist']}' (score: {canonical['score']:.1f}, song: {canonical.get('song_score', 0):.1f}, artist: {canonical.get('artist_score', 0):.1f})")
        else:
            # Rejected but show best match if available
            if best_match:
                algorithm = best_match.get('algorithm', 'Unknown')
                # Determine why it was rejected
                low_individual = (best_match.get('song_score', 0) < actual_threshold or 
                                best_match.get('artist_score', 0) < actual_threshold)
                low_combined = best_match.get('score', 0) < max(50, confidence_threshold - 15)
                
                reasons = []
                if low_individual:
                    if best_match.get('song_score', 0) < actual_threshold:
                        reasons.append(f"song score {best_match.get('song_score', 0):.1f} < {actual_threshold}")
                    if best_match.get('artist_score', 0) < actual_threshold:
                        reasons.append(f"artist score {best_match.get('artist_score', 0):.1f} < {actual_threshold}")
                if low_combined:
                    reasons.append(f"combined score {best_match.get('score', 0):.1f} < {max(50, confidence_threshold - 15)}")
                
                reason_text = ", ".join(reasons)
                log_print(f"⚠️ ({algorithm}) REJECTED '{song}' by '{artist}' - Best match: '{best_match['song']}' by '{best_match['artist']}' (score: {best_match['score']:.1f}, song: {best_match.get('song_score', 0):.1f}, artist: {best_match.get('artist_score', 0):.1f}) - Reason: {reason_text}")
            else:
                log_print(f"⚠️ Low confidence: '{song}' by '{artist}' (no viable matches found)")
        
        return canonical
    
    def save_caches():
        for filename, cache_data in [
            ('api_cache.pkl', api_cache),
            ('canonical_cache.pkl', canonical_cache),
            ('failed_queries.pkl', failed_queries)
        ]:
            try:
                with open(checkpoint_dir / filename, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                log_print(f"⚠️ Error saving {filename}: {e}")
    
    # Load and process data
    log_print(f"📁 Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Get unique combinations
    valid_mask = (
        df['Song'].notna() & df['Artist'].notna() & 
        (df['Song'].astype(str).str.strip() != '') & 
        (df['Artist'].astype(str).str.strip() != '')
    )
    unique_combos = df[valid_mask][['Song', 'Artist']].drop_duplicates()
    
    # Pre-filter if enabled
    if enable_pre_filter:
        log_print("🔍 Pre-filtering combinations...")
        filtered_combos = []
        for _, row in unique_combos.iterrows():
            song = str(row['Song']).lower()
            artist = str(row['Artist']).lower()
            
            # Skip obvious non-music entries
            non_music_keywords = ['chart', 'total', 'points', 'sales', 'streams', 'airplay', 'week']
            if any(keyword in song or keyword in artist for keyword in non_music_keywords):
                continue
            if len(song.strip()) < 2 or len(artist.strip()) < 2:
                continue
            if song.strip().isdigit() or artist.strip().isdigit():
                continue
            
            filtered_combos.append(row)
        
        unique_combos = pd.DataFrame(filtered_combos)
        log_print(f"📊 After filtering: {len(unique_combos)} combinations")
    
    log_print(f"🔄 Processing {len(unique_combos)} unique combinations...")
    
    # Process combinations
    successful_matches = 0
    failed_matches = 0
    
    for i, (_, row) in enumerate(unique_combos.iterrows()):
        song, artist = row['Song'], row['Artist']
        
        if (i + 1) % 10 == 0:
            log_print(f"[{i+1}/{len(unique_combos)}] Processing: '{song}' by '{artist}'")
        
        canonical = get_canonical_names(song, artist, individual_score_threshold)
        
        if canonical:
            successful_matches += 1
        else:
            failed_matches += 1
        
        # Save checkpoint periodically
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_caches()
            log_print(f"💾 Checkpoint: {i+1}/{len(unique_combos)} processed, {successful_matches} matches")
    
    # Final save
    save_caches()
    
    # Apply canonicalization to dataset
    log_print(f"🔄 Applying canonicalization to dataset...")
    df_result = df.copy()
    updated_rows = 0
    
    for idx, row in df.iterrows():
        try:
            song_val = row['Song']
            artist_val = row['Artist']
            
            # Convert to string and check for empty/null values
            song_str = str(song_val).strip()
            artist_str = str(artist_val).strip()
            
            # Skip if empty, null, or nan string representations
            if (song_str in ['', 'nan', 'None', 'NaN'] or 
                artist_str in ['', 'nan', 'None', 'NaN']):
                continue
                
        except Exception:
            continue  # Skip any problematic rows
        
        combo_hash = get_combo_hash(song_str, artist_str, individual_score_threshold)  # Use threshold-aware hash
        if combo_hash in canonical_cache and canonical_cache[combo_hash]:
            canonical = canonical_cache[combo_hash]
            df_result.at[idx, 'Song'] = canonical['song']
            df_result.at[idx, 'Artist'] = canonical['artist']
            updated_rows += 1
    
    # Save result
    df_result.to_csv(output_csv_path, index=False)
    
    log_print(f"\n📊 CANONICALIZATION SUMMARY")
    log_print(f"=" * 35)
    log_print(f"📊 Unique combinations: {len(unique_combos):,}")
    log_print(f"✅ Successful matches: {successful_matches:,}")
    log_print(f"❌ Failed matches: {failed_matches:,}")
    log_print(f"📝 Rows updated: {updated_rows:,}")
    log_print(f"📁 Output saved: {output_csv_path}")
    log_print(f"📁 Log saved: {log_file_path}")
    
    if successful_matches > 0:
        success_rate = (successful_matches / len(unique_combos)) * 100
        log_print(f"📈 Success rate: {success_rate:.1f}%")
    
    # Close logging handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    return df_result


async def run_async_musicbrainz_canonicalization(input_csv_path, output_csv_path, confidence_threshold=75,
                                               individual_score_threshold=75,
                                               enable_pre_filter=True, resume_from_checkpoint=True,
                                               concurrent_requests=3):
    """
    Async version of MusicBrainz canonicalization with optimized concurrent processing
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file
        confidence_threshold: Minimum confidence score for matches (default 75)
        individual_score_threshold: Minimum score required for both song and artist (default 75)
        enable_pre_filter: Whether to pre-filter non-music entries (default True)
        resume_from_checkpoint: Whether to resume from existing checkpoint (default True)
        concurrent_requests: Number of concurrent API requests (default 3, respects rate limits)
    
    Returns:
        DataFrame with canonicalized data
    """
    import asyncio
    import aiohttp
    import time
    from datetime import datetime
    import pickle
    from pathlib import Path
    import logging
    
    # Setup logging to file
    log_file_path = output_csv_path.replace('.csv', '_async_canonicalization.log')
    
    # Create a logger
    logger = logging.getLogger('async_musicbrainz_canonicalization')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    def log_print(message):
        """Helper function to log to both file and console"""
        logger.info(message)
    
    log_print("🚀 Starting Async MusicBrainz Canonicalization")
    log_print(f"📁 Input: {input_csv_path}")
    log_print(f"📁 Output: {output_csv_path}")
    log_print(f"📁 Log file: {log_file_path}")
    log_print(f"🎯 Confidence threshold: {confidence_threshold}%")
    log_print(f"🎵 Individual score threshold: {individual_score_threshold}%")
    log_print(f"⚡ Concurrent requests: {concurrent_requests}")
    
    # Configuration
    MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
    USER_AGENT = "ChartDataProcessor/1.0 (https://github.com/yourproject/charts)"
    RATE_LIMIT_DELAY = 1.2  # Slightly higher for async safety
    MAX_RETRIES = 3
    CHECKPOINT_INTERVAL = 50  # Less frequent for async (processes faster)
    
    # Setup checkpoint directory
    checkpoint_dir = Path("mb_checkpoints_async")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize caches
    api_cache = {}
    canonical_cache = {}
    failed_queries = set()
    
    # Load existing caches
    cache_files = {
        'api_cache.pkl': api_cache,
        'canonical_cache.pkl': canonical_cache,
        'failed_queries.pkl': failed_queries
    }
    
    for filename, cache_dict in cache_files.items():
        filepath = checkpoint_dir / filename
        if filepath.exists() and resume_from_checkpoint:
            try:
                with open(filepath, 'rb') as f:
                    loaded_data = pickle.load(f)
                    cache_dict.update(loaded_data)
                log_print(f"✅ Loaded cache {filename}: {len(loaded_data)} entries")
            except Exception as e:
                log_print(f"⚠️ Error loading {filename}: {e}")
    
    # Rate limiting semaphore
    rate_limiter = asyncio.Semaphore(concurrent_requests)
    
    def get_combo_hash(song, artist, threshold=None):
        import hashlib
        combo_str = f"{str(song).lower().strip()}|{str(artist).lower().strip()}"
        if threshold is not None:
            combo_str += f"|{threshold}"
        return hashlib.md5(combo_str.encode()).hexdigest()
    
    def normalize_for_search(text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).strip()
        text = re.sub(r'\b(feat\.?|featuring|ft\.?)\s+.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
        text = re.sub(r'\s*\[[^\]]*\]\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        special_chars = r'[+\-&|!(){}[\]^"~*?:\\]'
        text = re.sub(special_chars, r'\\\g<0>', text)
        return text
    
    def quick_confidence_check_async(data, song, artist, min_threshold=35):
        """Quick confidence check for async version"""
        if not data or 'recordings' not in data or not data['recordings']:
            return 0
        
        # Check the best result only for speed
        recording = data['recordings'][0]
        try:
            mb_song = recording.get('title', '')
            mb_artist = ''
            
            if 'artist-credit' in recording and recording['artist-credit']:
                artists = []
                for credit in recording['artist-credit']:
                    if isinstance(credit, dict) and 'artist' in credit:
                        artists.append(credit['artist'].get('name', ''))
                    elif isinstance(credit, str):
                        artists.append(credit)
                mb_artist = ', '.join(filter(None, artists))
            
            if not mb_song or not mb_artist:
                return 0
            
            # Quick similarity calculation
            if RAPIDFUZZ_AVAILABLE:
                song_score = calculate_rapidfuzz_similarity(song, mb_song)
                artist_score = calculate_rapidfuzz_similarity(artist, mb_artist)
            else:
                song_score = calculate_enhanced_similarity(song, mb_song)
                artist_score = calculate_enhanced_similarity(artist, mb_artist)
            
            # Return the average confidence
            return (song_score + artist_score) / 2
            
        except Exception:
            return 0
    
    async def query_musicbrainz_async(session, song, artist):
        """Async version of MusicBrainz query with rate limiting"""
        combo_hash = get_combo_hash(song, artist)
        
        if combo_hash in failed_queries:
            return None
        if combo_hash in api_cache:
            return api_cache[combo_hash]
        
        norm_song = normalize_for_search(song)
        norm_artist = normalize_for_search(artist)
        
        if not norm_song or not norm_artist:
            failed_queries.add(combo_hash)
            return None
        
        # Search strategies - prioritize partial matching for OCR data
        search_strategies = [
            f'recording:{norm_song} AND artist:{norm_artist}',  # Partial matching without quotes (DEFAULT)
            f'recording:"{norm_song}" AND artist:"{norm_artist}"',
            f'recording:"{norm_song}" OR artist:"{norm_artist}"',
            f'recording:{norm_song} OR artist:{norm_artist}',
            f'recording:"{norm_song}"',
            f'artist:"{norm_artist}"',
            f'recording:{norm_song}',  # Partial song search without quotes
            f'artist:{norm_artist}'    # Partial artist search without quotes
        ]
        
        params = {'limit': 50, 'fmt': 'json'}  # Increased from 20 to 50 for more thorough matching
        
        async with rate_limiter:  # Rate limiting
            for strategy_idx, query in enumerate(search_strategies):
                params['query'] = query
                
                for attempt in range(MAX_RETRIES):
                    try:
                        async with session.get(f"{MUSICBRAINZ_BASE_URL}/recording", 
                                             params=params, timeout=15) as response:
                            
                            if response.status == 200:
                                data = await response.json()
                                
                                if data.get('recordings') and len(data['recordings']) > 0:
                                    # For the first strategy (partial matching), accept any results
                                    # For other strategies, check confidence
                                    if strategy_idx == 0:
                                        # Always accept partial matching results (strategy 1)
                                        api_cache[combo_hash] = data
                                        await asyncio.sleep(RATE_LIMIT_DELAY)
                                        return data
                                    else:
                                        # Quick confidence check for other strategies
                                        confidence = quick_confidence_check_async(data, norm_song, norm_artist, min_threshold=35)
                                        
                                        if confidence >= 35 or strategy_idx >= len(search_strategies) - 1:
                                            # Good confidence or last strategy - cache and return
                                            api_cache[combo_hash] = data
                                            await asyncio.sleep(RATE_LIMIT_DELAY)
                                            return data
                                        elif strategy_idx < len(search_strategies) - 1:
                                            # Low confidence - try next strategy
                                            log_print(f"🔄 Strategy {strategy_idx + 1} confidence too low ({confidence:.1f}%), trying next strategy")
                                            break
                                elif strategy_idx < len(search_strategies) - 1:
                                    break
                                    
                            elif response.status == 429:
                                wait_time = 5 * (attempt + 1)
                                log_print(f"⏳ Rate limited, waiting {wait_time}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                if attempt == MAX_RETRIES - 1:
                                    break
                                await asyncio.sleep(RATE_LIMIT_DELAY)
                                
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            break
                        await asyncio.sleep(RATE_LIMIT_DELAY * (attempt + 1))
                
                await asyncio.sleep(RATE_LIMIT_DELAY / 2)  # Brief delay between strategies
        
        failed_queries.add(combo_hash)
        return None
    
    def calculate_enhanced_similarity(s1, s2):
        """Enhanced similarity calculation (same as sync version)"""
        s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
        
        if s1 == s2:
            return 100
        
        def normalize_for_comparison(text):
            text = re.sub(r'\b(feat\.?|featuring|ft\.?)\s+.*$', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*\([^)]*\)', '', text)
            text = re.sub(r'\s*\[[^\]]*\]', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        norm_s1 = normalize_for_comparison(s1)
        norm_s2 = normalize_for_comparison(s2)
        
        if norm_s1 == norm_s2:
            return 95
        
        # Simple Levenshtein similarity
        def levenshtein_similarity(a, b):
            if not a or not b:
                return 0
            if a == b:
                return 100
            
            max_len = max(len(a), len(b))
            if max_len == 0:
                return 100
            
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(a, b)
            similarity = ((max_len - distance) / max_len) * 100
            return max(0, similarity)
        
        levenshtein_score = levenshtein_similarity(norm_s1, norm_s2)
        
        # Weighted combination
        final_score = levenshtein_score * 0.8 + (50 if norm_s1 in norm_s2 or norm_s2 in norm_s1 else 0) * 0.2
        
        return min(100, max(0, final_score))
    
    def calculate_rapidfuzz_similarity(s1, s2):
        """RapidFuzz similarity calculation (same as sync version)"""
        if not RAPIDFUZZ_AVAILABLE:
            return calculate_enhanced_similarity(s1, s2)
        
        try:
            from rapidfuzz import fuzz
            
            s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
            
            if s1 == s2:
                return 100
            
            # Multiple RapidFuzz algorithms with weights
            ratio_score = fuzz.ratio(s1, s2)
            partial_ratio_score = fuzz.partial_ratio(s1, s2)
            token_sort_score = fuzz.token_sort_ratio(s1, s2)
            token_set_score = fuzz.token_set_ratio(s1, s2)
            wratio_score = fuzz.WRatio(s1, s2)
            
            # Weighted combination optimized for music data
            final_score = (
                wratio_score * 0.4 +           # Primary: Overall weighted ratio
                token_sort_score * 0.25 +      # Secondary: Token order invariant
                ratio_score * 0.2 +            # Tertiary: Direct character match
                partial_ratio_score * 0.1 +    # Quaternary: Substring match
                token_set_score * 0.05         # Quinary: Set-based match
            )
            
            return min(100, max(0, final_score))
            
        except Exception:
            return calculate_enhanced_similarity(s1, s2)
    
    def extract_canonical_names_optimized(mb_data, original_song, original_artist, individual_score_threshold=75):
        """Optimized version with smart early termination"""
        if not mb_data or 'recordings' not in mb_data:
            return None, None
        
        recordings = mb_data['recordings']
        if not recordings:
            return None, None
        
        best_match = None
        best_score = 0
        excellent_match_threshold = 90  # Lowered from 95 to 90 for more matches
        good_match_threshold = 75       # Lowered from 85 to 75 for more matches
        min_checks = 5                  # Increased from 3 to 5
        max_checks = 25                 # Increased from 12 to 25 for thorough comparison
        
        actual_checks = min(max_checks, len(recordings))
        
        for i, recording in enumerate(recordings[:actual_checks]):
            try:
                mb_song = recording.get('title', '')
                mb_artist = ''
                
                if 'artist-credit' in recording and recording['artist-credit']:
                    artists = []
                    for credit in recording['artist-credit']:
                        if isinstance(credit, dict) and 'artist' in credit:
                            artists.append(credit['artist'].get('name', ''))
                        elif isinstance(credit, str):
                            artists.append(credit)
                    mb_artist = ', '.join(filter(None, artists))
                
                if not mb_song or not mb_artist:
                    continue
                
                # Use RapidFuzz similarity if available
                if RAPIDFUZZ_AVAILABLE:
                    song_score = calculate_rapidfuzz_similarity(original_song, mb_song)
                    artist_score = calculate_rapidfuzz_similarity(original_artist, mb_artist)
                else:
                    song_score = calculate_enhanced_similarity(original_song, mb_song)
                    artist_score = calculate_enhanced_similarity(original_artist, mb_artist)
                
                # Enhanced weighted scoring
                if song_score == 100 and artist_score >= 85:
                    combined_score = 98
                elif artist_score == 100 and song_score >= 85:
                    combined_score = 95
                elif song_score >= 95 and artist_score >= 95:
                    combined_score = 96
                else:
                    combined_score = (song_score * 0.65 + artist_score * 0.35)
                
                # Track the best match
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        'song': mb_song,
                        'artist': mb_artist,
                        'score': combined_score,
                        'mbid': recording.get('id', ''),
                        'song_score': song_score,
                        'artist_score': artist_score,
                        'algorithm': 'RapidFuzz' if RAPIDFUZZ_AVAILABLE else 'Enhanced'
                    }
                
                # SMART EARLY TERMINATION - More conservative for better matches
                if i >= min_checks:  # Only after checking minimum results
                    if combined_score >= excellent_match_threshold and i >= min_checks * 2:
                        break  # Found excellent match after thorough check
                    elif combined_score >= good_match_threshold and i >= min_checks * 3:
                        break  # Found good match after extensive search
                        
            except Exception:
                continue
        
        # Check if the best match meets acceptance criteria
        if best_match:
            threshold_with_tolerance = max(50, confidence_threshold - 15)
            meets_individual_thresholds = (best_match['song_score'] >= individual_score_threshold and 
                                          best_match['artist_score'] >= individual_score_threshold)
            meets_combined_threshold = best_match['score'] >= threshold_with_tolerance
            
            if meets_individual_thresholds and meets_combined_threshold:
                return best_match, best_match
            else:
                return None, best_match
        
        return None, None
    
    async def get_canonical_names_async(session, song, artist, threshold=None):
        """Async version of canonical name extraction"""
        combo_hash_with_threshold = get_combo_hash(song, artist, threshold)
        
        if combo_hash_with_threshold in canonical_cache:
            cached_result = canonical_cache[combo_hash_with_threshold]
            if cached_result:
                algorithm = cached_result.get('algorithm', 'Unknown')
                log_print(f"💾 Cached ({algorithm}): '{song}' → '{cached_result['song']}' | '{artist}' → '{cached_result['artist']}' (score: {cached_result['score']:.1f})")
            return cached_result
        
        mb_data = await query_musicbrainz_async(session, song, artist)
        if not mb_data:
            canonical_cache[combo_hash_with_threshold] = None
            log_print(f"❌ No results: '{song}' by '{artist}'")
            return None
        
        canonical, best_match = extract_canonical_names_optimized(mb_data, song, artist, threshold or individual_score_threshold)
        canonical_cache[combo_hash_with_threshold] = canonical
        
        actual_threshold = threshold or individual_score_threshold
        
        if canonical:
            algorithm = canonical.get('algorithm', 'Unknown')
            log_print(f"✅ ({algorithm}) '{song}' → '{canonical['song']}' | '{artist}' → '{canonical['artist']}' (score: {canonical['score']:.1f}, song: {canonical.get('song_score', 0):.1f}, artist: {canonical.get('artist_score', 0):.1f})")
        else:
            if best_match:
                algorithm = best_match.get('algorithm', 'Unknown')
                low_individual = (best_match.get('song_score', 0) < actual_threshold or 
                                best_match.get('artist_score', 0) < actual_threshold)
                low_combined = best_match.get('score', 0) < max(50, confidence_threshold - 15)
                
                reasons = []
                if low_individual:
                    if best_match.get('song_score', 0) < actual_threshold:
                        reasons.append(f"song score {best_match.get('song_score', 0):.1f} < {actual_threshold}")
                    if best_match.get('artist_score', 0) < actual_threshold:
                        reasons.append(f"artist score {best_match.get('artist_score', 0):.1f} < {actual_threshold}")
                if low_combined:
                    reasons.append(f"combined score {best_match.get('score', 0):.1f} < {max(50, confidence_threshold - 15)}")
                
                reason_text = ", ".join(reasons)
                log_print(f"⚠️ ({algorithm}) REJECTED '{song}' by '{artist}' - Best match: '{best_match['song']}' by '{best_match['artist']}' (score: {best_match['score']:.1f}, song: {best_match.get('song_score', 0):.1f}, artist: {best_match.get('artist_score', 0):.1f}) - Reason: {reason_text}")
            else:
                log_print(f"⚠️ Low confidence: '{song}' by '{artist}' (no viable matches found)")
        
        return canonical
    
    async def save_caches_async():
        """Async-safe cache saving"""
        for filename, cache_data in [
            ('api_cache.pkl', api_cache),
            ('canonical_cache.pkl', canonical_cache),
            ('failed_queries.pkl', failed_queries)
        ]:
            try:
                with open(checkpoint_dir / filename, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                log_print(f"⚠️ Error saving {filename}: {e}")
    
    async def process_combinations_batch(session, combinations_batch):
        """Process a batch of combinations concurrently"""
        tasks = []
        for song, artist in combinations_batch:
            task = get_canonical_names_async(session, song, artist, individual_score_threshold)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    # Main async processing
    log_print(f"📁 Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Get unique combinations
    valid_mask = (
        df['Song'].notna() & df['Artist'].notna() & 
        (df['Song'].astype(str).str.strip() != '') & 
        (df['Artist'].astype(str).str.strip() != '')
    )
    unique_combos = df[valid_mask][['Song', 'Artist']].drop_duplicates()
    
    # Pre-filter if enabled
    if enable_pre_filter:
        log_print("🔍 Pre-filtering combinations...")
        filtered_combos = []
        for _, row in unique_combos.iterrows():
            song = str(row['Song']).lower()
            artist = str(row['Artist']).lower()
            
            non_music_keywords = ['chart', 'total', 'points', 'sales', 'streams', 'airplay', 'week']
            if any(keyword in song or keyword in artist for keyword in non_music_keywords):
                continue
            if len(song.strip()) < 2 or len(artist.strip()) < 2:
                continue
            if song.strip().isdigit() or artist.strip().isdigit():
                continue
            
            filtered_combos.append(row)
        
        unique_combos = pd.DataFrame(filtered_combos)
        log_print(f"📊 After filtering: {len(unique_combos)} combinations")
    
    log_print(f"🚀 Processing {len(unique_combos)} unique combinations with async processing...")
    
    # Process combinations in batches
    batch_size = concurrent_requests * 5  # Process in larger batches for efficiency
    successful_matches = 0
    failed_matches = 0
    
    async with aiohttp.ClientSession(
        headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'},
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        
        combinations_list = [(row['Song'], row['Artist']) for _, row in unique_combos.iterrows()]
        
        for i in range(0, len(combinations_list), batch_size):
            batch = combinations_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(combinations_list) + batch_size - 1) // batch_size
            
            log_print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch)} combinations...")
            
            batch_start_time = time.time()
            results = await process_combinations_batch(session, batch)
            batch_end_time = time.time()
            
            # Count results
            for result in results:
                if isinstance(result, Exception):
                    failed_matches += 1
                elif result:
                    successful_matches += 1
                else:
                    failed_matches += 1
            
            log_print(f"[Batch {batch_num}/{total_batches}] Completed in {batch_end_time - batch_start_time:.1f}s - {successful_matches} total matches so far")
            
            # Save checkpoint periodically
            if batch_num % (CHECKPOINT_INTERVAL // batch_size + 1) == 0:
                await save_caches_async()
                log_print(f"💾 Checkpoint: {i + len(batch)}/{len(combinations_list)} processed, {successful_matches} matches")
    
    # Final save
    await save_caches_async()
    
    # Apply canonicalization to dataset
    log_print(f"🔄 Applying canonicalization to dataset...")
    df_result = df.copy()
    updated_rows = 0
    
    for idx, row in df.iterrows():
        try:
            song_val = row['Song']
            artist_val = row['Artist']
            
            song_str = str(song_val).strip()
            artist_str = str(artist_val).strip()
            
            if (song_str in ['', 'nan', 'None', 'NaN'] or 
                artist_str in ['', 'nan', 'None', 'NaN']):
                continue
                
        except Exception:
            continue
        
        combo_hash = get_combo_hash(song_str, artist_str, individual_score_threshold)
        if combo_hash in canonical_cache and canonical_cache[combo_hash]:
            canonical = canonical_cache[combo_hash]
            df_result.at[idx, 'Song'] = canonical['song']
            df_result.at[idx, 'Artist'] = canonical['artist']
            updated_rows += 1
    
    # Save result
    df_result.to_csv(output_csv_path, index=False)
    
    log_print(f"\n📊 ASYNC CANONICALIZATION SUMMARY")
    log_print(f"=" * 40)
    log_print(f"📊 Unique combinations: {len(unique_combos):,}")
    log_print(f"✅ Successful matches: {successful_matches:,}")
    log_print(f"❌ Failed matches: {failed_matches:,}")
    log_print(f"📝 Rows updated: {updated_rows:,}")
    log_print(f"📁 Output saved: {output_csv_path}")
    log_print(f"📁 Log saved: {log_file_path}")
    log_print(f"⚡ Concurrent requests used: {concurrent_requests}")
    
    if successful_matches > 0:
        success_rate = (successful_matches / len(unique_combos)) * 100
        log_print(f"📈 Success rate: {success_rate:.1f}%")
    
    # Close logging handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    return df_result


def run_musicbrainz_canonicalization_colab_async(input_csv_path="final_merged_all_dates.csv", 
                                                 output_csv_path=None,
                                                 confidence_threshold=75,
                                                 individual_score_threshold=75,
                                                 enable_pre_filter=True,
                                                 resume_from_checkpoint=True,
                                                 concurrent_requests=3):
    """
    Wrapper function to run async MusicBrainz canonicalization in Google Colab
    
    Args:
        input_csv_path: Path to input CSV file (default: "final_merged_all_dates.csv")
        output_csv_path: Path to output CSV file (optional, auto-generated if None)
        confidence_threshold: Minimum confidence score for matches (default 75)
        individual_score_threshold: Minimum score required for both song and artist (default 75)
        enable_pre_filter: Whether to pre-filter non-music entries (default True)
        resume_from_checkpoint: Whether to resume from existing checkpoint (default True)
        concurrent_requests: Number of concurrent API requests (1-5, default 3)
    
    Returns:
        Path to canonical CSV file or None if failed
    """
    import asyncio
    
    print("🚀 Async MusicBrainz Canonicalization for Google Colab")
    print("=" * 60)
    
    # Check if running in Colab
    is_colab = check_colab_environment()
    
    # Install required packages if needed
    print("\n📦 Checking dependencies...")
    try:
        import aiohttp
        print("✅ aiohttp available")
    except ImportError:
        print("⚠️ Installing aiohttp...")
        import subprocess
        subprocess.run(['pip', 'install', 'aiohttp'], check=True)
        import aiohttp
        print("✅ aiohttp installed")
    
    try:
        import requests
        print("✅ requests available")
    except ImportError:
        print("⚠️ Installing requests...")
        import subprocess
        subprocess.run(['pip', 'install', 'requests'], check=True)
        import requests
        print("✅ requests installed")
    
    # Check if input file exists
    if not os.path.exists(input_csv_path):
        print(f"❌ Input file not found: {input_csv_path}")
        print("Available files in current directory:")
        for file in os.listdir("."):
            if file.endswith('.csv'):
                print(f"   📄 {file}")
        return None
    
    # Auto-generate output path if not provided
    if output_csv_path is None:
        base_name = os.path.splitext(input_csv_path)[0]
        output_csv_path = f"{base_name}_canonical_async.csv"
    
    # Validate concurrent_requests
    concurrent_requests = max(1, min(5, concurrent_requests))  # Clamp between 1-5
    
    print(f"\n📁 Input file: {input_csv_path}")
    print(f"📁 Output file: {output_csv_path}")
    print(f"🎯 Confidence threshold: {confidence_threshold}%")
    print(f"🎵 Individual score threshold: {individual_score_threshold}% (both song and artist must meet this)")
    print(f"🔍 Pre-filtering: {'Enabled' if enable_pre_filter else 'Disabled'}")
    print(f"💾 Resume from checkpoint: {'Yes' if resume_from_checkpoint else 'No'}")
    print(f"⚡ Concurrent requests: {concurrent_requests} (respects rate limits)")
    
    try:
        # Run the async canonicalization
        print("\n🔧 Setting up async MusicBrainz canonicalizer...")
        
        # Run in event loop
        if hasattr(asyncio, 'get_running_loop'):
            try:
                loop = asyncio.get_running_loop()
                # We're in an existing event loop (like Jupyter), create a task
                import nest_asyncio
                nest_asyncio.apply()
                canonical_df = asyncio.run(run_async_musicbrainz_canonicalization(
                    input_csv_path=input_csv_path,
                    output_csv_path=output_csv_path,
                    confidence_threshold=confidence_threshold,
                    individual_score_threshold=individual_score_threshold,
                    enable_pre_filter=enable_pre_filter,
                    resume_from_checkpoint=resume_from_checkpoint,
                    concurrent_requests=concurrent_requests
                ))
            except RuntimeError:
                # No existing loop, create new one
                canonical_df = asyncio.run(run_async_musicbrainz_canonicalization(
                    input_csv_path=input_csv_path,
                    output_csv_path=output_csv_path,
                    confidence_threshold=confidence_threshold,
                    individual_score_threshold=individual_score_threshold,
                    enable_pre_filter=enable_pre_filter,
                    resume_from_checkpoint=resume_from_checkpoint,
                    concurrent_requests=concurrent_requests
                ))
        else:
            # Older Python versions
            canonical_df = asyncio.run(run_async_musicbrainz_canonicalization(
                input_csv_path=input_csv_path,
                output_csv_path=output_csv_path,
                confidence_threshold=confidence_threshold,
                individual_score_threshold=individual_score_threshold,
                enable_pre_filter=enable_pre_filter,
                resume_from_checkpoint=resume_from_checkpoint,
                concurrent_requests=concurrent_requests
            ))
        
        if canonical_df is not None:
            # Save Excel version
            xlsx_path = output_csv_path.replace('.csv', '.xlsx')
            try:
                canonical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
                print(f"✅ Excel version saved: {xlsx_path}")
            except Exception as e:
                print(f"⚠️ Could not save Excel version: {e}")
            
            # Save to Google Drive if mounted
            if os.path.exists("/content/drive"):
                print("\n☁️ Saving to Google Drive...")
                save_to_drive(output_csv_path, f"twitter_scraper/{os.path.basename(output_csv_path)}")
                if os.path.exists(xlsx_path):
                    save_to_drive(xlsx_path, f"twitter_scraper/{os.path.basename(xlsx_path)}")
            
            # Download files in Colab
            if is_colab:
                print(f"\n⬇️ Downloading canonical files...")
                download_file(output_csv_path)
                if os.path.exists(xlsx_path):
                    download_file(xlsx_path)
            
            print(f"\n🎉 Async MusicBrainz canonicalization completed successfully!")
            print(f"📊 Canonical dataset: {len(canonical_df)} rows")
            print(f"⚡ Performance: Async processing with {concurrent_requests} concurrent requests")
            return output_csv_path
        else:
            print(f"❌ Async canonicalization failed")
            return None
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(f"💡 Install missing packages: !pip install aiohttp nest-asyncio")
        return None
    except Exception as e:
        print(f"❌ Error during async canonicalization: {e}")
        print(f"💡 Tip: Try the synchronous version if async fails")
        print(f"💡 Tip: Reduce concurrent_requests if getting rate limit errors")
        return None
def print_colab_instructions():
    """Print instructions for using this script in Google Colab"""
    print("""
🚀 Google Colab OCR Batch Processing Instructions (GPU Optimized + Fuzzy Standardization)
=========================================================================================

1. SETUP GPU RUNTIME:
   
   # IMPORTANT: Enable GPU runtime in Colab!
   # Go to Runtime > Change runtime type > Hardware accelerator > GPU (T4/V100/A100)
   
   # Verify GPU is available
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")

2. INSTALL DEPENDENCIES:
   
   # Install with CUDA support for GPU acceleration + fuzzy matching + async support
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install doctr[torch] pandas pillow python-dateutil openpyxl rapidfuzz aiohttp nest-asyncio
   
   # Mount Google Drive (optional)
   from google.colab import drive
   drive.mount('/content/drive')

3. UPLOAD FILES:
   
   # Upload your OCR.py file (recommended for full functionality)
   from google.colab import files
   uploaded = files.upload()
   
   # Upload your images as a zip file
   uploaded = files.upload()
   
   # Extract zip file
   import zipfile
   for filename in uploaded.keys():
       with zipfile.ZipFile(filename, 'r') as zip_ref:
           zip_ref.extractall('/content/')

4. RUN OCR PROCESSING:
   
   # Run the main OCR processing function
   result = run_ocr_batch_processing_colab()

5. RUN MUSICBRAINZ CANONICALIZATION (In a separate cell):
   
   # OPTION 1: SYNCHRONOUS (Standard, reliable)
   # Basic usage - canonicalize the final merged dataset
   canonical_path = run_musicbrainz_canonicalization_colab()
   
   # Advanced usage with custom settings
   canonical_path = run_musicbrainz_canonicalization_colab(
       input_csv_path="final_merged_all_dates.csv",
       confidence_threshold=90,          # Higher confidence = fewer but more accurate matches
       individual_score_threshold=75,    # Both song AND artist must meet this threshold
       enable_pre_filter=True,           # Filter out non-music entries
       resume_from_checkpoint=True       # Resume from previous run if interrupted
   )
   
   # OPTION 2: ASYNCHRONOUS (Faster, concurrent processing)
   # Basic async usage - faster processing with concurrent requests
   canonical_path = run_musicbrainz_canonicalization_colab_async()
   
   # Advanced async usage with custom settings
   canonical_path = run_musicbrainz_canonicalization_colab_async(
       input_csv_path="final_merged_all_dates.csv",
       confidence_threshold=90,          # Higher confidence = fewer but more accurate matches
       individual_score_threshold=75,    # Both song AND artist must meet this threshold
       enable_pre_filter=True,           # Filter out non-music entries
       resume_from_checkpoint=True,      # Resume from previous run if interrupted
       concurrent_requests=3             # Number of concurrent API requests (1-5)
   )
   
   # Process a specific file (sync version)
   canonical_path = run_musicbrainz_canonicalization_colab(
       input_csv_path="my_custom_data.csv",
       output_csv_path="my_canonical_data.csv"
   )
   
   # Process a specific file (async version)
   canonical_path = run_musicbrainz_canonicalization_colab_async(
       input_csv_path="my_custom_data.csv",
       output_csv_path="my_canonical_data_async.csv",
       concurrent_requests=5             # More aggressive concurrent processing
   )

6. EXPECTED DIRECTORY STRUCTURE:
   /content/
   ├── sorted_images/
   │   ├── Aug 25/
   │   │   ├── 2025-08-10_1.jpg
   │   │   ├── 2025-08-10_2.jpg
   │   │   └── ...
   │   └── Aug 26/
   │       └── ...
   ├── chart_dfs/          (created automatically)
   │   ├── final_merged_all_dates.csv
   │   └── final_merged_all_dates.xlsx
   ├── mb_checkpoints/     (MusicBrainz cache - sync version)
   │   ├── api_cache.pkl
   │   ├── canonical_cache.pkl
   │   └── failed_queries.pkl
   ├── mb_checkpoints_async/  (MusicBrainz cache - async version)
   │   ├── api_cache.pkl
   │   ├── canonical_cache.pkl
   │   └── failed_queries.pkl
   ├── final_merged_all_dates.csv
   ├── final_merged_all_dates_canonical.csv      (sync output)
   ├── final_merged_all_dates_canonical_async.csv (async output)
   ├── final_merged_all_dates_canonical.xlsx
   └── final_merged_all_dates_canonical_async.xlsx

7. PERFORMANCE NOTES:
   - ⚡ GPU processing is 3-10x faster than CPU for OCR
   - 🔥 Script automatically detects and uses GPU if available
   - 💾 Memory is automatically managed with cache clearing
   - 📊 Progress monitoring shows device being used
   - ⏱️ Processing times are displayed for performance tracking
   - 📁 Final results saved in both CSV and XLSX formats
   - 📅 Dates automatically formatted as mm/dd/yy (American format)
   - 🧹 Text data automatically normalized (lowercase, punctuation removal, accent normalization)
   - 🔢 Numeric values in song/artist names automatically moved to change column
   - 🆕 FUZZY STANDARDIZATION using RapidFuzz for consistent spelling
   - 🎵 MUSICBRAINZ CANONICALIZATION for authoritative song/artist names
   - ☁️ Files automatically uploaded to Google Drive if mounted
   - ⬇️ Both formats automatically downloaded for local use

8. MUSICBRAINZ CANONICALIZATION FEATURES:
   - 💾 Automatic checkpoint saving and resuming from interruptions
   - 🔄 Smart caching to avoid redundant API calls  
   - ⏱️ Rate limiting to respect MusicBrainz API (1+ requests/second)
   - 🔍 Pre-filtering to remove non-music entries
   - 🎯 Confidence scoring for match quality
   - 📊 Detailed progress and statistics reporting
   - 🌐 Queries MusicBrainz database for authoritative names
   - ⚡ ASYNC VERSION: Concurrent processing for 2-3x speed improvement
   - 🎛️ SYNC VERSION: Reliable, sequential processing
   - 📈 Expected processing times:
     * SYNC: 1,000 unique songs: ~17 minutes (first run), ~2 minutes (cached)
     * ASYNC: 1,000 unique songs: ~8 minutes (first run), ~1 minute (cached)
     * SYNC: 5,000 unique songs: ~1.4 hours (first run), ~8 minutes (cached)
     * ASYNC: 5,000 unique songs: ~35 minutes (first run), ~4 minutes (cached)
     * SYNC: 10,000 unique songs: ~2.8 hours (first run), ~15 minutes (cached)
     * ASYNC: 10,000 unique songs: ~1.2 hours (first run), ~8 minutes (cached)

9. DATA CLEANING FEATURES:
   - Text normalization: lowercase, punctuation removal, whitespace cleanup
   - Accent normalization: é→e, ñ→n, ü→u, etc.
   - Leading "=" sign removal from text fields
   - Music-specific normalization:
     * Standardizes featuring credits: "featuring" → "feat.", "ft." → "feat."
     * Standardizes conjunctions: "&" → "and"
     * Removes remix/edit info: "(Remix)", "(Radio Edit)", "(Live)", "(Acoustic)"
   - Automatic extraction of change values from song/artist text (e.g., "5 Song Name" → Song: "song name", Change: "+5")
   - Only extracts changes when Change column is empty (preserves existing values)
   - 🆕 FUZZY STANDARDIZATION using RapidFuzz:
     * Standardizes spelling variations WITHOUT reducing dataset size
     * Example: "Don't Stop Believin'" and "Dont Stop Believing" both become "Don't Stop Believin'"
     * Handles artist variations: "Bon Jovi" vs "BonJovi" → both become "Bon Jovi"
     * Preserves ALL original rows and data (ranks, dates, points, etc.)
     * Only changes spelling for consistency
   - 🆕 MUSICBRAINZ CANONICALIZATION:
     * Provides authoritative song and artist names from MusicBrainz database
     * Example: "Bohemian Rhapsody" variations → "Bohemian Rhapsody" (official)
     * Artist name standardization: "Queen" variations → "Queen" (official)
     * Includes MusicBrainz IDs for future reference and linking

10. FUZZY STANDARDIZATION DETAILS:
   - 🎯 Combo Threshold: 85% - for full song-artist combinations
   - 🎤 Artist Threshold: 80% - for artist name variations  
   - 🎵 Song Threshold: 85% - for song title variations
   - 📊 Shows unique combination reduction (not row reduction)
   - 🔍 Handles common OCR errors and spelling variations
   - ✅ Keeps dataset size exactly the same
   - 🎯 Only standardizes spelling for better analysis consistency

11. EXPECTED OUTPUT:
   - Same number of rows as input (no data loss)
   - Consistent spelling for song-artist combinations (fuzzy standardization)
   - Authoritative names from MusicBrainz database (canonicalization)
   - Better data quality for analysis and aggregation
   - Detailed processing reports with before/after statistics

12. BENEFITS FOR ANALYSIS:
    - More accurate aggregation (sales totals, chart performance)
    - Better trend analysis across time periods
    - Consistent artist/song identification
    - Reduced noise in data visualization
    - Improved matching with external databases
    - Authoritative names for professional analysis

13. TROUBLESHOOTING:
   - If GPU not detected: Change runtime type to GPU
   - For memory errors: Restart runtime and clear cache
   - For import errors: Reinstall packages with correct CUDA version
   - For date format issues: Re-run processing (dates now use mm/dd/yy format)
   - For text data issues: Text normalization is applied automatically
   - For RapidFuzz errors: !pip install rapidfuzz
   - For memory issues with large datasets: Process in smaller chunks
   - For slow fuzzy matching: Increase thresholds to be less aggressive
   - Dataset too large: Standardization works on chunks automatically
   - For MusicBrainz errors: Check internet connection, reduce confidence threshold
   - For interrupted MusicBrainz runs: Simply run the same command again to resume
   - For async errors: Install missing packages: !pip install aiohttp nest-asyncio
   - For async rate limiting: Reduce concurrent_requests parameter (try 1-2)
   - For async timeout errors: Check internet connection, try sync version
   - Choose sync vs async: Async is faster but sync is more reliable for unstable connections

14. SYNC VS ASYNC COMPARISON:
   ✅ SYNCHRONOUS VERSION (run_musicbrainz_canonicalization_colab):
   - More reliable and stable
   - Better error handling
   - Easier debugging
   - Sequential processing (1 request at a time)
   - Use when: Unstable internet, debugging needed, first-time users
   
   ⚡ ASYNCHRONOUS VERSION (run_musicbrainz_canonicalization_colab_async):
   - 2-3x faster processing
   - Concurrent API requests (respects rate limits)
   - Smart early termination for optimal results
   - Advanced batch processing
   - Use when: Large datasets, stable internet, experienced users
   - Configurable concurrent_requests (1-5, default 3)""")

if __name__ == "__main__":
    # Print instructions
    print_colab_instructions()
    
    # Test DocTR setup
    print("\n🔧 Testing DocTR setup...")
    if setup_doctr():
        print("✅ DocTR is ready to use!")
        print(f"✅ OCR model initialized: {ocr_model is not None}")
        print(f"✅ DocumentFile available: {DocumentFile is not None}")
        
        # Check if model is on GPU
        try:
            import torch
            if ocr_model is not None and hasattr(ocr_model, 'device'):
                print(f"🔥 Model device: {next(ocr_model.parameters()).device}")
            elif torch.cuda.is_available():
                print("🔥 GPU available for processing")
        except:
            pass
    else:
        print("❌ DocTR setup failed. Please install:")
        print("!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("!pip install doctr[torch]")
    
    # Uncomment the next line to run the OCR processing
    # result = run_ocr_batch_processing_colab()
    
    # After OCR processing is complete, run MusicBrainz canonicalization in a separate cell:
    # canonical_path = run_musicbrainz_canonicalization_colab()
    #
    # Advanced usage with custom settings:
    # canonical_path = run_musicbrainz_canonicalization_colab(
    #     input_csv_path="final_merged_all_dates.csv",
    #     confidence_threshold=90,           # Higher confidence = fewer but more accurate matches
    #     individual_score_threshold=75,     # Both song AND artist must meet this threshold
    #     enable_pre_filter=True,            # Filter out non-music entries  
    #     resume_from_checkpoint=True        # Resume from previous run if interrupted
    # )
    #
    # 💡 CACHE REUSE: If you want to try different thresholds, the script will reuse 
    # API data and only re-evaluate the scoring. This means:
    # - Changing individual_score_threshold: Uses cached API data, no new API calls
    # - Changing confidence_threshold: Uses cached API data, no new API calls  
    # - First run downloads data from MusicBrainz API and caches it
    # - Subsequent runs with different thresholds are much faster!


def demonstrate_threshold_cache_reuse():
    """
    Example function showing how to test different thresholds without re-fetching API data.
    This function demonstrates the cache reuse functionality.
    """
    print("🧪 Testing Cache Reuse with Different Thresholds")
    print("=" * 50)
    
    test_input = "final_merged_all_dates.csv"
    
    # First run: Downloads data from API and caches it
    print("\n🔄 First run (threshold=75) - Will fetch from API")
    result1 = run_musicbrainz_canonicalization_colab(
        input_csv_path=test_input,
        output_csv_path="canonical_threshold_75.csv", 
        individual_score_threshold=75,
        resume_from_checkpoint=True
    )
    
    # Second run: Reuses cached API data with stricter threshold
    print("\n🔄 Second run (threshold=85) - Will reuse cached API data")
    result2 = run_musicbrainz_canonicalization_colab(
        input_csv_path=test_input,
        output_csv_path="canonical_threshold_85.csv",
        individual_score_threshold=85,  # Stricter threshold
        resume_from_checkpoint=True
    )
    
    # Third run: Reuses cached API data with more lenient threshold  
    print("\n🔄 Third run (threshold=65) - Will reuse cached API data")
    result3 = run_musicbrainz_canonicalization_colab(
        input_csv_path=test_input,
        output_csv_path="canonical_threshold_65.csv",
        individual_score_threshold=65,  # More lenient threshold
        resume_from_checkpoint=True
    )
    
    print("\n✅ Cache reuse test completed!")
    print("📊 Notice how runs 2 and 3 were much faster - they reused cached API data")
    print("🎯 Each run produced different results based on the threshold, but without new API calls")
    
    return [result1, result2, result3]


# Uncomment the line below to test cache reuse behavior:
# demonstrate_threshold_cache_reuse()
