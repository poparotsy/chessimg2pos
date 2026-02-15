#!/usr/bin/env python3
"""
Prepare all downloaded datasets - extract tiles from images with FEN in filename
"""
import os
import sys
import glob
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessboard_image import get_chessboard_tiles

def extract_fen_from_filename(filename):
    """Extract FEN from filename (format: FEN.jpg or FEN.jpeg)"""
    # Remove extension
    fen = os.path.splitext(filename)[0]
    # Replace - with /
    fen = fen.replace('-', '/')
    return fen

def process_dataset(images_dir, output_dir, dataset_name):
    """Process a dataset where FEN is encoded in filename"""
    print(f"\n📦 Processing {dataset_name}...")
    print(f"   Source: {images_dir}")
    print(f"   Output: {output_dir}")
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if not image_files:
        print(f"   ❌ No images found")
        return 0
    
    print(f"   Found {len(image_files)} images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    success = 0
    failed = 0
    files = "abcdefgh"
    
    for img_path in image_files:
        if success % 100 == 0 and success > 0:
            print(f"   Processed {success}...")
        
        filename = os.path.basename(img_path)
        
        try:
            # Extract FEN from filename
            fen_board = extract_fen_from_filename(filename)
            rows = fen_board.split("/")
            
            if len(rows) != 8:
                failed += 1
                continue
            
            # Extract tiles
            tiles = get_chessboard_tiles(img_path, use_grayscale=True)
            if len(tiles) != 64:
                failed += 1
                continue
            
            # Create subdirectory
            board_id = f"{dataset_name}_{success:05d}"
            sub_dir = os.path.join(output_dir, board_id)
            os.makedirs(sub_dir, exist_ok=True)
            
            # Save tiles
            tile_idx = 0
            for row_idx, row_str in enumerate(rows):
                col_idx = 0
                for char in row_str:
                    if char.isdigit():
                        for _ in range(int(char)):
                            sqr = f"{files[col_idx]}{8-row_idx}"
                            tiles[tile_idx].save(os.path.join(sub_dir, f"{sqr}_1.png"))
                            tile_idx += 1
                            col_idx += 1
                    else:
                        sqr = f"{files[col_idx]}{8-row_idx}"
                        tiles[tile_idx].save(os.path.join(sub_dir, f"{sqr}_{char}.png"))
                        tile_idx += 1
                        col_idx += 1
            
            success += 1
            
        except Exception as e:
            failed += 1
            if failed < 5:
                print(f"   ⚠️  {filename}: {e}")
    
    print(f"   ✅ Success: {success}")
    print(f"   ❌ Failed: {failed}")
    return success

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(base_dir, "training_images")
    output_dir = os.path.join(base_dir, "images", "tiles_real")
    
    print("\n🎯 Prepare Real Chess Datasets")
    print("=" * 50)
    
    total_processed = 0
    
    # 1. GitHub dataset (500 images)
    github_dir = os.path.join(training_dir, "github_chess", "labeled_originals")
    if os.path.exists(github_dir):
        count = process_dataset(github_dir, output_dir, "github")
        total_processed += count
    
    # 2. Kaggle/GTS dataset
    kaggle_train = os.path.join(training_dir, "kaggle_gts", "train")
    if os.path.exists(kaggle_train):
        count = process_dataset(kaggle_train, output_dir, "kaggle")
        total_processed += count
    
    # 3. OSF dataset (need to extract first)
    osf_dir = os.path.join(training_dir, "osf_chess")
    if os.path.exists(osf_dir):
        print(f"\n⚠️  OSF dataset needs manual extraction")
        print(f"   Location: {osf_dir}")
        print(f"   Extract train.zip and rerun this script")
    
    print()
    print("=" * 50)
    print(f"📊 Total tiles extracted: {total_processed} boards")
    print(f"📁 Tiles saved to: {output_dir}")
    
    if total_processed > 0:
        print()
        print("Next step:")
        print("  python3 train_finetune_real.py")
    else:
        print()
        print("❌ No data processed. Check that datasets are extracted.")

if __name__ == "__main__":
    main()
