#!/usr/bin/env python3
"""
Generate tiles from ALL chessboard images, handling any filename format
"""
import os
import glob
import math
from PIL import Image
import sys

# Add src to path to import local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chessimg2pos.chessboard_image import get_chessboard_tiles

base_dir = os.path.dirname(os.path.abspath(__file__))
chessboards_dir = os.path.join(base_dir, "images", "chessboards")
tiles_dir = os.path.join(base_dir, "images", "tiles_all")

print(f"📁 Source: {chessboards_dir}")
print(f"📁 Target: {tiles_dir}")

# Create tiles directory
os.makedirs(tiles_dir, exist_ok=True)

# Find all PNG files recursively
all_images = glob.glob(os.path.join(chessboards_dir, "**", "*.png"), recursive=True)
print(f"📊 Found {len(all_images)} images")

success = 0
failed = 0
files = "abcdefgh"

for idx, img_path in enumerate(all_images):
    if idx % 100 == 0:
        print(f"Processing {idx}/{len(all_images)}...")
    
    try:
        # Get tiles from image
        tiles = get_chessboard_tiles(img_path, use_grayscale=True)
        
        if len(tiles) != 64:
            failed += 1
            continue
        
        # Parse filename to get piece positions
        filename = os.path.basename(img_path)
        fen_part = filename[:-4]  # Remove .png
        
        # Try to parse as FEN format (rows separated by -)
        rows = fen_part.split("-")
        
        if len(rows) != 8:
            # Skip files that don't have proper FEN format
            failed += 1
            continue
        
        # Create subdirectory for this board
        sub_dir = os.path.join(tiles_dir, fen_part)
        os.makedirs(sub_dir, exist_ok=True)
        
        # Save each tile with piece label
        tile_idx = 0
        for row_idx in range(8):
            row = rows[row_idx]
            for col_idx in range(8):
                if col_idx < len(row):
                    piece = row[col_idx]
                else:
                    # If row is shorter, treat as empty
                    piece = '1'
                
                # Square name (a1, b2, etc.)
                sqr_id = f"{files[col_idx]}{8 - row_idx}"
                
                # Save tile
                tile_filename = os.path.join(sub_dir, f"{sqr_id}_{piece}.png")
                tiles[tile_idx].save(tile_filename, format="PNG")
                tile_idx += 1
        
        success += 1
            
    except Exception as e:
        failed += 1
        if failed < 5:  # Show first few errors
            print(f"  ⚠️  Failed on {os.path.basename(img_path)}: {e}")

print(f"\n✅ Successfully processed: {success}")
print(f"❌ Failed: {failed}")
print(f"📁 Tiles saved to: {tiles_dir}")

if success > 0:
    print(f"\n🚀 Now run training with:")
    print(f"   python3 -c \"from chessimg2pos import ChessRecognitionTrainer; t = ChessRecognitionTrainer('{tiles_dir}', './models/model_ultra_all.pt', generate_tiles=False, epochs=15); t.train(classifier='ultra')\"")
