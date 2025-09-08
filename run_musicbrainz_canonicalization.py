# run_musicbrainz_canonicalization.py - Simple runner for MusicBrainz canonicalization
import os
import sys
from pathlib import Path

def run_musicbrainz_canonicalization(input_csv, output_csv=None, resume=True, pre_filter=True):
    """
    Simple function to run MusicBrainz canonicalization from the main pipeline
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional)
        resume: Whether to resume from checkpoint
        pre_filter: Whether to pre-filter non-music entries
    
    Returns:
        Path to canonical CSV file
    """
    try:
        from musicbrainz_canonicalization import MusicBrainzCanonicalizer
        
        print("\n🎵 Starting MusicBrainz Canonicalization...")
        print("=" * 50)
        
        # Initialize canonicalizer
        canonicalizer = MusicBrainzCanonicalizer(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            checkpoint_dir="mb_checkpoints"
        )
        
        # Process dataset
        result_df = canonicalizer.process_dataset(
            pre_filter=pre_filter,
            resume_from_checkpoint=resume
        )
        
        print(f"\n✅ MusicBrainz canonicalization completed!")
        print(f"📁 Canonical dataset saved: {canonicalizer.output_csv_path}")
        
        return str(canonicalizer.output_csv_path)
        
    except ImportError:
        print("❌ MusicBrainz canonicalization module not found")
        print("Please ensure musicbrainz_canonicalization.py is in the same directory")
        return None
    except Exception as e:
        print(f"❌ Error during MusicBrainz canonicalization: {e}")
        return None

def add_musicbrainz_to_pipeline(final_df, csv_path, enable_musicbrainz=False):
    """
    Add MusicBrainz canonicalization as final step in OCR pipeline
    
    Args:
        final_df: Final processed DataFrame
        csv_path: Path where CSV was saved
        enable_musicbrainz: Whether to run MusicBrainz canonicalization
    
    Returns:
        Path to canonical CSV or original CSV path
    """
    if not enable_musicbrainz:
        print("⚠️ MusicBrainz canonicalization disabled")
        return csv_path
        
    if final_df is None or final_df.empty:
        print("⚠️ No data to canonicalize")
        return csv_path
        
    print(f"\n🎵 Running MusicBrainz canonicalization on {len(final_df)} rows...")
    
    canonical_csv = run_musicbrainz_canonicalization(
        input_csv=csv_path,
        output_csv=csv_path.replace('.csv', '_canonical.csv'),
        resume=True,
        pre_filter=True
    )
    
    return canonical_csv if canonical_csv else csv_path

if __name__ == "__main__":
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python run_musicbrainz_canonicalization.py <input_csv> [output_csv]")
        print("Example: python run_musicbrainz_canonicalization.py final_merged_all_dates.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    result = run_musicbrainz_canonicalization(input_file, output_file)
    
    if result:
        print(f"\n🎉 Success! Canonical dataset: {result}")
    else:
        print(f"\n❌ Canonicalization failed!")
        sys.exit(1)
