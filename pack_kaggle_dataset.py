#!/usr/bin/env python3
"""
📦 TENSOR PACKER BEAST
- Converts 5M+ PNG tiles into 10 optimized .pt tensor files.
- Uses all CPU cores for parallel processing.
- Saves as uint8 to minimize disk and RAM usage (5GB total).
"""
import os
import glob
import time
import torch
import numpy as np
from PIL import Image
import concurrent.futures
from datetime import timedelta

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
CHUNKS = 10
NUM_CPU_WORKERS = os.cpu_count() or 4

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = os.path.join(base_dir, "images", "tiles_real")
if not os.path.exists(tiles_dir):
    tiles_dir = os.path.abspath(os.path.join(base_dir, "..", "images", "tiles_real"))

output_dir = os.path.join(base_dir, "tensor_dataset")
os.makedirs(output_dir, exist_ok=True)

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def process_single_image(path):
    piece_type = path[-5]
    if piece_type not in FEN_CHARS:
        return None
    try:
        img = Image.open(path).convert("L" if USE_GRAYSCALE else "RGB")
        img = img.resize((32, 32))
        return np.array(img, dtype=np.uint8), FEN_CHARS.index(piece_type)
    except:
        return None

def pack_chunk(chunk_idx, paths):
    print(f"📦 Processing Chunk {chunk_idx+1}/{CHUNKS} ({len(paths):,} images)...")
    start = time.time()
    
    imgs = []
    lbls = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
        results = list(executor.map(process_single_image, paths))
        
    for res in results:
        if res is not None:
            imgs.append(res[0])
            lbls.append(res[1])
            
    if not imgs:
        print(f"⚠️ Chunk {chunk_idx+1} is empty!")
        return

    # Convert to Tensors
    x = torch.from_numpy(np.stack(imgs)) # (N, 32, 32) uint8
    if USE_GRAYSCALE:
        x = x.unsqueeze(1) # (N, 1, 32, 32)
    else:
        x = x.permute(0, 3, 1, 2) # (N, 3, 32, 32)
        
    y = torch.tensor(lbls, dtype=torch.uint8)
    
    chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:02d}.pt")
    torch.save({'x': x, 'y': y}, chunk_path)
    
    elapsed = time.time() - start
    print(f"✅ Saved {chunk_path} | {len(imgs):,} images | {elapsed:.1f}s")

def main():
    print("🚀 TENSOR PACKER BEAST - INITIALIZING")
    print("="*50)
    
    print(f"🔍 Scanning {tiles_dir}...")
    all_paths = []
    board_dirs = glob.glob(os.path.join(tiles_dir, "*"))
    for d in board_dirs:
        if os.path.isdir(d):
            all_paths.extend(glob.glob(os.path.join(d, "*.png")))
    
    total_images = len(all_paths)
    if total_images == 0:
        print("❌ No images found!")
        return
        
    print(f"📊 Found {total_images:,} images. Splitting into {CHUNKS} chunks.")
    np.random.seed(42)
    np.random.shuffle(all_paths)
    
    chunk_size = (total_images + CHUNKS - 1) // CHUNKS
    
    start_total = time.time()
    for i in range(CHUNKS):
        chunk_paths = all_paths[i*chunk_size : (i+1)*chunk_size]
        if not chunk_paths: break
        pack_chunk(i, chunk_paths)
        
    print("="*50)
    print(f"🎉 PACKING COMPLETE! Total Time: {format_time(time.time() - start_total)}")
    print(f"📁 Tensors saved in: {output_dir}")

if __name__ == "__main__":
    main()
