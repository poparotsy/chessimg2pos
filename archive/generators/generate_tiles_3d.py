#!/usr/bin/env python3
"""
Generate tiles from the 3D rendered chess dataset
Dataset: datasets.chess with train/val/test splits
"""
import os
import json
import sys
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessboard_image import get_chessboard_tiles

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "datasets.chess")
tiles_dir = os.path.join(base_dir, "images", "tiles_3d")

os.makedirs(tiles_dir, exist_ok=True)

success = 0
failed = 0
files = "abcdefgh"

# Process train, val, and test sets
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        continue
    
    print(f"\n📂 Processing {split} set...")
    
    # Find all JSON files
    json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        img_id = json_file[:-5]  # Remove .json
        img_path = os.path.join(split_dir, f"{img_id}.png")
        json_path = os.path.join(split_dir, json_file)
        
        if not os.path.exists(img_path):
            failed += 1
            continue
        
        try:
            # Load FEN from JSON
            with open(json_path) as f:
                data = json.load(f)
                fen_full = data['fen']
            
            # Extract tiles
            tiles = get_chessboard_tiles(img_path, use_grayscale=True)
            if len(tiles) != 64:
                failed += 1
                continue
            
            fen_board = fen_full.split()[0] if ' ' in fen_full else fen_full
            rows = fen_board.split("/")
            
            sub_dir = os.path.join(tiles_dir, f"{split}_{img_id}")
            os.makedirs(sub_dir, exist_ok=True)
            
            tile_idx = 0
            for row_idx, row in enumerate(rows):
                col_idx = 0
                for char in row:
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
            if success % 100 == 0:
                print(f"  Processed {success}...")
                
        except Exception as e:
            failed += 1
            if failed < 5:
                print(f"  ⚠️  {img_id}: {e}")

print(f"\n✅ Success: {success}")
print(f"❌ Failed: {failed}")
print(f"📁 Tiles: {tiles_dir}")
print(f"\n🚀 Now run: python3 train_3d_rendered.py")
