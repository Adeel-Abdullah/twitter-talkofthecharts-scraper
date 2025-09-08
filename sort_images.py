import os
import re
from datetime import datetime
import shutil

def sort_image_by_date(filename, source_dir="charts_images", base_target_dir="sorted_images"):
    """
    Sort an image file into a directory based on its date.
    
    Args:
        filename (str): The filename like '2023-08-11_1.jpg'
        source_dir (str): Directory where the file currently is
        base_target_dir (str): Base directory for sorted files
    
    Returns:
        str: Path where the file was moved, or None if failed
    """
    # Extract date from filename (format: YYYY-MM-DD_X.jpg)
    date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})_\d+\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
    
    if not date_match:
        print(f"[sort_image] Filename '{filename}' doesn't match expected format (YYYY-MM-DD_X.jpg)")
        return None
    
    year, month, day = date_match.groups()[:3]
    
    try:
        # Parse the date to get month name
        date_obj = datetime(int(year), int(month), int(day))
        month_name = date_obj.strftime("%b")  # Short month name (Jan, Feb, etc.)
        year_short = date_obj.strftime("%y")  # Short year (23, 24, etc.)
        
        # Create directory name: "Aug 23", "Sep 24", etc.
        dir_name = f"{month_name} {year_short}"
        target_dir = os.path.join(base_target_dir, dir_name)
        
        # Create directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        print(f"[sort_image] Created/verified directory: {target_dir}")
        
        # Source and destination paths
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"[sort_image] Source file not found: {source_path}")
            return None
        
        # Check if target file already exists
        if os.path.exists(target_path):
            print(f"[sort_image] File already exists in target location: {target_path}")
            return target_path
        
        # Move the file
        shutil.move(source_path, target_path)
        print(f"[sort_image] Moved: {source_path} -> {target_path}")
        
        return target_path
        
    except Exception as e:
        print(f"[sort_image] Error processing {filename}: {e}")
        return None

def sort_all_images(source_dir="charts_images", base_target_dir="sorted_images"):
    """
    Sort all images in the source directory into month-based subdirectories.
    
    Args:
        source_dir (str): Directory containing images to sort
        base_target_dir (str): Base directory for sorted files
    """
    if not os.path.exists(source_dir):
        print(f"[sort_all] Source directory not found: {source_dir}")
        return
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    files = [f for f in os.listdir(source_dir) 
             if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(source_dir, f))]
    
    if not files:
        print(f"[sort_all] No image files found in {source_dir}")
        return
    
    print(f"[sort_all] Found {len(files)} image files to sort")
    
    success_count = 0
    for filename in files:
        result = sort_image_by_date(filename, source_dir, base_target_dir)
        if result:
            success_count += 1
    
    print(f"[sort_all] Successfully sorted {success_count}/{len(files)} files")

if __name__ == "__main__":
    # Example usage
    print("=== Image Sorting Script ===")
    
    # Sort all images in the charts_images directory
    sort_all_images()
    
    # Or sort a specific file
    # sort_image_by_date("2023-08-11_1.jpg")
