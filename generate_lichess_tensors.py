#!/usr/bin/env python3
"""
🎨 LICHESS TENSOR GENERATOR (BEAST MODE)
- Generates synthetic chess puzzles directly into .pt tensors.
- Randomizes board colors and piece distributions.
- Zero disk I/O bottleneck: renders in RAM and saves chunks.
- Required: pip install python-chess cairosvg
"""
import os
import time
import torch
import numpy as np
import random
import chess
import chess.svg
from PIL import Image
from io import BytesIO
import concurrent.futures
from datetime import timedelta
import argparse

try:
    import cairosvg
except ImportError:
    print("❌ Missing cairosvg. Install with: pip install cairosvg")
    exit(1)

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
IMG_SIZE = 32 # Standard tile size for our models
CHUNKS_TRAIN = 8
CHUNKS_VAL = 2
BOARDS_PER_CHUNK = 5000 # Total 50,000 boards = 3.2M tiles

# Lichess-style themes (Light, Dark)
THEMES = [
    ("#f0d9b5", "#b58863"), # Wood
    ("#eeeeee", "#8ca2ad"), # Blue-Grey
    ("#ffffff", "#769656"), # Green
    ("#dee3e6", "#8ca2ad"), # Marble
    ("#ffffdd", "#86a666"), # Classic
]

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "tensor_dataset_synthetic")
os.makedirs(output_dir, exist_ok=True)

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def render_board_to_tiles(fen, theme):
    """Renders a FEN to a list of 64 numpy tiles"""
    board = chess.Board(fen)
    # Render SVG
    svg_data = chess.svg.board(
        board, 
        colors={'square light': theme[0], 'square dark': theme[1]},
        size=400 # High res for clean slicing
    )
    
    # SVG -> PNG -> PIL
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    full_img = Image.open(BytesIO(png_data)).convert("L") # Grayscale
    
    # Slice into 64 tiles (each 50x50 at 400 total size)
    w, h = full_img.size
    ts = w // 8
    tiles = []
    labels = []
    
    # python-chess FEN order is rank 8 to 1 (top to bottom)
    for row in range(8):
        for col in range(8):
            left = col * ts
            top = row * ts
            tile = full_img.crop((left, top, left + ts, top + ts))
            tile = tile.resize((IMG_SIZE, IMG_SIZE))
            
            # Get label
            sq = chess.square(col, 7 - row)
            piece = board.piece_at(sq)
            if piece is None:
                label = 0 # '1'
            else:
                label = FEN_CHARS.index(piece.symbol())
                
            tiles.append(np.array(tile, dtype=np.uint8))
            labels.append(label)
            
    return tiles, labels

def generate_random_fen():
    """Generates a random but realistic-ish board"""
    board = chess.Board(None)
    # Place Kings
    board.set_piece_at(random.choice(list(chess.SQUARES)), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(random.choice([s for s in chess.SQUARES if board.piece_at(s) is None]), chess.Piece(chess.KING, chess.BLACK))
    
    # Place 5-20 random pieces
    for _ in range(random.randint(5, 25)):
        sqs = [s for s in chess.SQUARES if board.piece_at(s) is None]
        if not sqs: break
        pt = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        c = random.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(random.choice(sqs), chess.Piece(pt, c))
    return board.fen()

def process_chunk(name):
    print(f"🎨 Generating {name}...")
    start = time.time()
    
    all_imgs = []
    all_lbls = []
    
    for i in range(BOARDS_PER_CHUNK):
        fen = generate_random_fen()
        theme = random.choice(THEMES)
        tiles, labels = render_board_to_tiles(fen, theme)
        all_imgs.extend(tiles)
        all_lbls.extend(labels)
        
        if (i+1) % 500 == 0:
            print(f"   {name}: {i+1}/{BOARDS_PER_CHUNK} boards done...")

    x = torch.from_numpy(np.stack(all_imgs)).unsqueeze(1) # (N, 1, 32, 32)
    y = torch.tensor(all_lbls, dtype=torch.uint8)
    
    path = os.path.join(output_dir, f"{name}.pt")
    torch.save({'x': x, 'y': y}, path)
    print(f"✅ Saved {path} | {len(all_imgs):,} tiles | {time.time()-start:.1f}s")

def main():
    print("🚀 LICHESS TENSOR GENERATOR - STARTING")
    print(f"📊 Target: {(CHUNKS_TRAIN + CHUNKS_VAL) * BOARDS_PER_CHUNK:,} boards")
    print(f"📂 Output: {output_dir}")
    
    start_total = time.time()
    
    # Generate Validation
    for i in range(CHUNKS_VAL):
        process_chunk(f"val_chunk_{i:02d}")
        
    # Generate Training
    for i in range(CHUNKS_TRAIN):
        process_chunk(f"train_chunk_{i:02d}")
        
    print("" + "="*60)
    print(f"🎉 SYNTHETIC DATA READY! Total Time: {format_time(time.time() - start_total)}")
    print(f"📁 Tensors saved in: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
