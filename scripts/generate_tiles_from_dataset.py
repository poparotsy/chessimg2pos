#!/usr/bin/env python3
import os
import json
import sys
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from chessimg2pos.chessboard_image import get_chessboard_tiles

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, "data", "train")
fen_file = os.path.join(base_dir, "data", "labels", "train_fen.json")
tiles_dir = os.path.join(base_dir, "images", "tiles_dataset")

os.makedirs(tiles_dir, exist_ok=True)

with open(fen_file) as f:
    fens = json.load(f)

print(f"📊 Processing {len(fens)} images...")

success = 0
failed = 0
files = "abcdefgh"

for img_id, fen_full in fens.items():  # Process all images
    if success % 1000 == 0:
        print(f"Processed {success}...")
    
    img_path = os.path.join(train_dir, f"CV_{img_id.zfill(7)}.jpg")
    if not os.path.exists(img_path):
        failed += 1
        continue
    
    try:
        tiles = get_chessboard_tiles(img_path, use_grayscale=True)
        if len(tiles) != 64:
            failed += 1
            continue
        
        fen_board = fen_full.split()[0]
        rows = fen_board.split("/")
        
        sub_dir = os.path.join(tiles_dir, img_id)
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
    except Exception as e:
        failed += 1
        if failed < 5:
            print(f"  ⚠️  {img_id}: {e}")

print(f"\n✅ Success: {success}")
print(f"❌ Failed: {failed}")
print(f"📁 Tiles: {tiles_dir}")
