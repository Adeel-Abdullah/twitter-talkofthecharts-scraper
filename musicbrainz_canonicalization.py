# musicbrainz_canonicalization.py - MusicBrainz API Canonicalization with Checkpoints
import pandas as pd
import requests
import time
import json
import os
import re
from datetime import datetime
import pickle
from pathlib import Path
import hashlib
from rapidfuzz import fuzz

# Configuration
MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
USER_AGENT = "ChartDataProcessor/1.0 (https://github.com/yourproject/charts)"
RATE_LIMIT_DELAY = 1.1  # Slightly over 1 second to be safe
MAX_RETRIES = 3
CONFIDENCE_THRESHOLD = 85  # Minimum confidence score to accept a match
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N processed items

class MusicBrainzCanonicalizer:
    def __init__(self, input_csv_path, output_csv_path=None, checkpoint_dir="mb_checkpoints"):
        """
        Initialize MusicBrainz canonicalizer with checkpoint support
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to output CSV file (optional)
            checkpoint_dir: Directory to store checkpoint files
        """
        self.input_csv_path = Path(input_csv_path)
        self.output_csv_path = Path(output_csv_path) if output_csv_path else self.input_csv_path.with_suffix('.canonical.csv')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize caches
        self.api_cache = {}
        self.canonical_cache = {}
        self.failed_queries = set()
        
        # Load existing caches and checkpoints
        self._load_caches()
        self._setup_session()
        
        # Statistics
        self.stats = {
            'total_unique_combos': 0,
            'processed_combos': 0,
            'api_hits': 0,
            'cache_hits': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'rows_updated': 0,
            'start_time': None,
            'checkpoint_saves': 0
        }
        
    def _setup_session(self):
        """Setup requests session with proper headers"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'application/json'
        })
        
    def _load_caches(self):
        """Load existing caches from checkpoint files"""
        cache_files = {
            'api_cache.pkl': 'api_cache',
            'canonical_cache.pkl': 'canonical_cache', 
            'failed_queries.pkl': 'failed_queries'
        }
        
        for filename, attr_name in cache_files.items():
            filepath = self.checkpoint_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
                    print(f"✅ Loaded {attr_name}: {len(getattr(self, attr_name))} entries")
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
                    
    def _save_caches(self):
        """Save caches to checkpoint files"""
        cache_data = {
            'api_cache.pkl': self.api_cache,
            'canonical_cache.pkl': self.canonical_cache,
            'failed_queries.pkl': self.failed_queries
        }
        
        for filename, data in cache_data.items():
            filepath = self.checkpoint_dir / filename
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"⚠️ Error saving {filename}: {e}")
                
    def _get_combo_hash(self, song, artist):
        """Generate unique hash for song-artist combination"""
        combo_str = f"{str(song).lower().strip()}|{str(artist).lower().strip()}"
        return hashlib.md5(combo_str.encode()).hexdigest()
        
    def _normalize_for_search(self, text):
        """Normalize text for MusicBrainz search"""
        if pd.isna(text) or text == '':
            return ''
            
        text = str(text).strip()
        
        # Remove common prefixes/suffixes that might interfere
        text = re.sub(r'\b(feat\.?|featuring|ft\.?)\s+.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)  # Remove parenthetical content
        text = re.sub(r'\s*\[[^\]]*\]\s*', ' ', text)  # Remove bracketed content
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Escape special characters for Lucene query
        special_chars = r'[+\-&|!(){}[\]^"~*?:\\]'
        text = re.sub(special_chars, r'\\\g<0>', text)
        
        return text
        
    def _query_musicbrainz_recording(self, song, artist, max_retries=MAX_RETRIES):
        """Query MusicBrainz for recording information"""
        combo_hash = self._get_combo_hash(song, artist)
        
        # Check if already queried and failed
        if combo_hash in self.failed_queries:
            return None
            
        # Check API cache
        if combo_hash in self.api_cache:
            self.stats['cache_hits'] += 1
            return self.api_cache[combo_hash]
            
        # Normalize search terms
        norm_song = self._normalize_for_search(song)
        norm_artist = self._normalize_for_search(artist)
        
        if not norm_song or not norm_artist:
            self.failed_queries.add(combo_hash)
            return None
            
        # Construct search query
        query = f'recording:"{norm_song}" AND artist:"{norm_artist}"'
        
        params = {
            'query': query,
            'limit': 5,
            'fmt': 'json'
        }
        
        for attempt in range(max_retries):
            try:
                print(f"🔍 Querying: '{song}' by '{artist}' (attempt {attempt + 1})")
                
                response = self.session.get(
                    f"{MUSICBRAINZ_BASE_URL}/recording",
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.api_cache[combo_hash] = data
                    self.stats['api_hits'] += 1
                    
                    # Rate limiting
                    time.sleep(RATE_LIMIT_DELAY)
                    return data
                    
                elif response.status_code == 429:  # Rate limited
                    wait_time = 5 * (attempt + 1)
                    print(f"⏳ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"⚠️ HTTP {response.status_code}: {response.text[:100]}")
                    if attempt == max_retries - 1:
                        self.failed_queries.add(combo_hash)
                    time.sleep(RATE_LIMIT_DELAY)
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error: {e}")
                if attempt == max_retries - 1:
                    self.failed_queries.add(combo_hash)
                time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
                
        return None
        
    def _extract_canonical_names(self, mb_data, original_song, original_artist):
        """Extract canonical names from MusicBrainz response"""
        if not mb_data or 'recordings' not in mb_data:
            return None
            
        recordings = mb_data['recordings']
        if not recordings:
            return None
            
        best_match = None
        best_score = 0
        
        for recording in recordings[:3]:  # Check top 3 results
            try:
                mb_song = recording.get('title', '')
                mb_artist = ''
                
                # Extract primary artist
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
                    
                # Calculate similarity scores
                song_score = fuzz.ratio(original_song.lower(), mb_song.lower())
                artist_score = fuzz.ratio(original_artist.lower(), mb_artist.lower())
                combined_score = (song_score + artist_score) / 2
                
                if combined_score > best_score and combined_score >= CONFIDENCE_THRESHOLD:
                    best_score = combined_score
                    best_match = {
                        'song': mb_song,
                        'artist': mb_artist,
                        'score': combined_score,
                        'mbid': recording.get('id', '')
                    }
                    
            except Exception as e:
                print(f"⚠️ Error processing recording: {e}")
                continue
                
        return best_match
        
    def _get_canonical_names(self, song, artist):
        """Get canonical names for song-artist combination"""
        combo_hash = self._get_combo_hash(song, artist)
        
        # Check canonical cache
        if combo_hash in self.canonical_cache:
            return self.canonical_cache[combo_hash]
            
        # Query MusicBrainz
        mb_data = self._query_musicbrainz_recording(song, artist)
        if not mb_data:
            self.canonical_cache[combo_hash] = None
            self.stats['failed_matches'] += 1
            return None
            
        # Extract canonical names
        canonical = self._extract_canonical_names(mb_data, song, artist)
        self.canonical_cache[combo_hash] = canonical
        
        if canonical:
            self.stats['successful_matches'] += 1
            print(f"✅ Match: '{song}' → '{canonical['song']}' | '{artist}' → '{canonical['artist']}' (score: {canonical['score']:.1f})")
        else:
            self.stats['failed_matches'] += 1
            print(f"❌ No match found for '{song}' by '{artist}'")
            
        return canonical
        
    def _save_checkpoint(self, processed_combos, total_combos):
        """Save processing checkpoint"""
        checkpoint_data = {
            'processed_combos': processed_combos,
            'total_combos': total_combos,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            self._save_caches()
            self.stats['checkpoint_saves'] += 1
            print(f"💾 Checkpoint saved: {processed_combos}/{total_combos} combinations processed")
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {e}")
            
    def _load_checkpoint(self):
        """Load processing checkpoint"""
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                print(f"📂 Found checkpoint: {data['processed_combos']}/{data['total_combos']} combinations processed")
                print(f"📅 Last checkpoint: {data['timestamp']}")
                return data
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
        return None
        
    def _get_unique_combinations(self, df):
        """Get unique song-artist combinations for processing"""
        # Filter out empty/invalid entries
        valid_mask = (
            df['Song'].notna() & 
            df['Artist'].notna() & 
            (df['Song'].astype(str).str.strip() != '') & 
            (df['Artist'].astype(str).str.strip() != '')
        )
        
        valid_df = df[valid_mask]
        unique_combos = valid_df[['Song', 'Artist']].drop_duplicates()
        
        print(f"📊 Found {len(unique_combos)} unique song-artist combinations")
        print(f"📊 Total rows in dataset: {len(df)}")
        print(f"📊 Valid combinations: {len(valid_df)} rows")
        
        return unique_combos
        
    def _pre_filter_combinations(self, unique_combos):
        """Pre-filter combinations to reduce API calls"""
        print("\n🔍 Pre-filtering combinations...")
        
        # Filter out obviously non-music entries
        music_keywords = ['song', 'track', 'single', 'album', 'feat', 'ft', 'remix']
        non_music_keywords = ['chart', 'total', 'points', 'sales', 'streams', 'airplay', 'week']
        
        filtered_combos = []
        
        for _, row in unique_combos.iterrows():
            song = str(row['Song']).lower()
            artist = str(row['Artist']).lower()
            
            # Skip if contains non-music keywords
            if any(keyword in song or keyword in artist for keyword in non_music_keywords):
                continue
                
            # Skip very short entries (likely OCR errors)
            if len(song.strip()) < 2 or len(artist.strip()) < 2:
                continue
                
            # Skip numeric-only entries
            if song.strip().isdigit() or artist.strip().isdigit():
                continue
                
            filtered_combos.append(row)
        
        filtered_df = pd.DataFrame(filtered_combos)
        print(f"📊 After filtering: {len(filtered_df)} combinations remain")
        print(f"📊 Filtered out: {len(unique_combos) - len(filtered_df)} combinations")
        
        return filtered_df
        
    def process_dataset(self, pre_filter=True, resume_from_checkpoint=True):
        """Process the dataset with MusicBrainz canonicalization"""
        print("🎵 Starting MusicBrainz Canonicalization")
        print("=" * 50)
        
        # Load input data
        print(f"📁 Loading data from: {self.input_csv_path}")
        df = pd.read_csv(self.input_csv_path)
        
        # Get unique combinations
        unique_combos = self._get_unique_combinations(df)
        
        # Pre-filter if requested
        if pre_filter:
            unique_combos = self._pre_filter_combinations(unique_combos)
            
        self.stats['total_unique_combos'] = len(unique_combos)
        self.stats['start_time'] = datetime.now()
        
        # Check for checkpoint
        checkpoint_data = None
        start_index = 0
        
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint()
            if checkpoint_data:
                start_index = checkpoint_data['processed_combos']
                self.stats.update(checkpoint_data['stats'])
                print(f"🔄 Resuming from checkpoint: starting at combination {start_index}")
        
        # Process combinations
        print(f"\n🔄 Processing {len(unique_combos)} unique combinations...")
        print(f"📊 Starting from index: {start_index}")
        print(f"📊 Cache status: {len(self.canonical_cache)} canonical entries, {len(self.api_cache)} API responses")
        
        processing_start_time = time.time()
        
        for i, (_, row) in enumerate(unique_combos.iterrows()):
            if i < start_index:
                continue
                
            song = row['Song']
            artist = row['Artist']
            
            print(f"\n[{i+1}/{len(unique_combos)}] Processing: '{song}' by '{artist}'")
            
            # Get canonical names
            canonical = self._get_canonical_names(song, artist)
            
            self.stats['processed_combos'] = i + 1
            
            # Save checkpoint periodically
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(i + 1, len(unique_combos))
                
                # Show progress statistics
                elapsed = time.time() - processing_start_time
                rate = (i + 1 - start_index) / elapsed if elapsed > 0 else 0
                remaining = len(unique_combos) - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                
                print(f"\n📊 Progress Report:")
                print(f"   ⏱️ Elapsed: {elapsed/60:.1f} minutes")
                print(f"   🎯 Rate: {rate:.2f} combinations/second")
                print(f"   ⏳ ETA: {eta/60:.1f} minutes")
                print(f"   ✅ Successful matches: {self.stats['successful_matches']}")
                print(f"   ❌ Failed matches: {self.stats['failed_matches']}")
                print(f"   💾 Cache hits: {self.stats['cache_hits']}")
                print(f"   🌐 API hits: {self.stats['api_hits']}")
        
        # Final checkpoint
        self._save_checkpoint(len(unique_combos), len(unique_combos))
        
        # Apply canonicalization to full dataset
        print(f"\n🔄 Applying canonicalization to full dataset...")
        df_canonical = self._apply_canonicalization_to_dataset(df)
        
        # Save results
        print(f"💾 Saving canonical dataset to: {self.output_csv_path}")
        df_canonical.to_csv(self.output_csv_path, index=False)
        
        # Save final statistics
        self._save_final_report()
        
        print(f"\n✅ Canonicalization completed!")
        self._print_final_statistics()
        
        return df_canonical
        
    def _apply_canonicalization_to_dataset(self, df):
        """Apply canonical names to the full dataset"""
        df_result = df.copy()
        updated_rows = 0
        
        print(f"🔄 Updating {len(df)} rows with canonical names...")
        
        for idx, row in df.iterrows():
            if pd.isna(row['Song']) or pd.isna(row['Artist']):
                continue
                
            combo_hash = self._get_combo_hash(row['Song'], row['Artist'])
            
            if combo_hash in self.canonical_cache and self.canonical_cache[combo_hash]:
                canonical = self.canonical_cache[combo_hash]
                df_result.at[idx, 'Song'] = canonical['song']
                df_result.at[idx, 'Artist'] = canonical['artist']
                
                # Add MusicBrainz ID if desired
                if 'MBID' not in df_result.columns:
                    df_result['MBID'] = ''
                df_result.at[idx, 'MBID'] = canonical['mbid']
                
                updated_rows += 1
                
        self.stats['rows_updated'] = updated_rows
        print(f"✅ Updated {updated_rows} rows with canonical names")
        
        return df_result
        
    def _save_final_report(self):
        """Save final processing report"""
        report = {
            'processing_summary': self.stats,
            'cache_sizes': {
                'canonical_cache': len(self.canonical_cache),
                'api_cache': len(self.api_cache),
                'failed_queries': len(self.failed_queries)
            },
            'files': {
                'input_file': str(self.input_csv_path),
                'output_file': str(self.output_csv_path)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = self.checkpoint_dir / 'final_report.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📋 Final report saved: {report_file}")
        except Exception as e:
            print(f"⚠️ Error saving final report: {e}")
            
    def _print_final_statistics(self):
        """Print final processing statistics"""
        total_time = (datetime.now() - datetime.fromisoformat(self.stats['start_time'].replace('Z', '+00:00'))).total_seconds()
        
        print(f"\n📊 FINAL STATISTICS")
        print(f"=" * 40)
        print(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
        print(f"📊 Total unique combinations: {self.stats['total_unique_combos']:,}")
        print(f"🔄 Combinations processed: {self.stats['processed_combos']:,}")
        print(f"✅ Successful matches: {self.stats['successful_matches']:,}")
        print(f"❌ Failed matches: {self.stats['failed_matches']:,}")
        print(f"📝 Rows updated: {self.stats['rows_updated']:,}")
        print(f"🌐 API requests made: {self.stats['api_hits']:,}")
        print(f"💾 Cache hits: {self.stats['cache_hits']:,}")
        print(f"💾 Checkpoints saved: {self.stats['checkpoint_saves']:,}")
        
        if self.stats['successful_matches'] > 0:
            success_rate = (self.stats['successful_matches'] / self.stats['processed_combos']) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
            
        if self.stats['api_hits'] > 0:
            cache_efficiency = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['api_hits'])) * 100
            print(f"⚡ Cache efficiency: {cache_efficiency:.1f}%")


def main():
    """Main function to run MusicBrainz canonicalization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MusicBrainz Canonicalization with Checkpoints')
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    parser.add_argument('--checkpoint-dir', '-c', default='mb_checkpoints', help='Checkpoint directory')
    parser.add_argument('--no-filter', action='store_true', help='Skip pre-filtering')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh (ignore checkpoints)')
    parser.add_argument('--confidence', type=int, default=85, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Update global confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence
    
    # Initialize canonicalizer
    canonicalizer = MusicBrainzCanonicalizer(
        input_csv_path=args.input_csv,
        output_csv_path=args.output,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Process dataset
    result_df = canonicalizer.process_dataset(
        pre_filter=not args.no_filter,
        resume_from_checkpoint=not args.no_resume
    )
    
    return result_df


if __name__ == "__main__":
    # Example usage
    print("🎵 MusicBrainz Canonicalization Script")
    print("=" * 40)
    print("\nUsage examples:")
    print("python musicbrainz_canonicalization.py final_merged_all_dates.csv")
    print("python musicbrainz_canonicalization.py input.csv --output canonical.csv")
    print("python musicbrainz_canonicalization.py input.csv --checkpoint-dir my_checkpoints")
    print("python musicbrainz_canonicalization.py input.csv --confidence 90 --no-filter")
    print("\nFeatures:")
    print("✅ Automatic checkpoint saving and resuming")
    print("✅ API response caching to avoid redundant queries")
    print("✅ Rate limiting (1 request/second) to respect MusicBrainz")
    print("✅ Pre-filtering to reduce unnecessary API calls")
    print("✅ Confidence scoring using fuzzy matching")
    print("✅ Detailed progress reporting and statistics")
    print("✅ Error handling and retry logic")
    print("✅ MusicBrainz ID preservation for future reference")
    
    # Run main function if called with arguments
    import sys
    if len(sys.argv) > 1:
        main()
