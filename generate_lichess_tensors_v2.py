#!/usr/bin/env python3
"""
🎨 LICHESS TENSOR GENERATOR (BEAST MODE v2 - REAL-WORLD SIM)
- Generates synthetic chess puzzles directly into .pt tensors.
- Randomizes board colors, piece sets, and tile sizes.
- Simulates real-world image artifacts:
    * JPEG compression (quality 30-95)
    * Gaussian blur (simulate rescaling artifacts)
    * Slight tile misalignment crops
    * Brightness / contrast jitter
- Zero disk I/O bottleneck: renders in RAM and saves chunks.
- Board coordinate labels (files a-h, ranks 1-8) can be added randomly,
  always, or never – controlled by COORDS_MODE.
- Required: pip install python-chess pillow torch torchvision
"""
import os
import io
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from datetime import timedelta
import random
import chess

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
IMG_SIZE = 32  # Standard tile size for our models
CHUNKS_TRAIN = 10   # 10 * 2000 = 20,000 boards → 1.28M tiles
CHUNKS_VAL   = 2    # 2 * 2000 = 4,000 boards → 256k tiles
BOARDS_PER_CHUNK = 2000

# Lichess / Chess.com style themes (Light square, Dark square)
THEMES = [
    ("#f0d9b5", "#b58863"),  # Lichess wood (brown)
    ("#eeeeee", "#8ca2ad"),  # Lichess blue-grey
    ("#ffffff", "#769656"),  # Lichess green
    ("#dee3e6", "#8ca2ad"),  # Marble
    ("#ffffdd", "#86a666"),  # Classic yellow-green
    ("#ffffff", "#58ac8a"),  # Chess.com green
    ("#f0e9c5", "#c5a55a"),  # Chess.com brown
    ("#e8ebef", "#7fa650"),  # Chesstempo green
    ("#cccccc", "#999999"),  # Greyscale (hardest!)
    ("#f9f6f2", "#b8836f"),  # Lichess newspaper
    ("#d9e4f0", "#6d8fa7"),  # Blue steel
    ("#faf0d7", "#8b6343"),  # Dark wood
]

# Coordinate mode: 'random' (each board decides), 'always', or 'never'
COORDS_MODE = 'random'

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "tensor_dataset_synthetic")
os.makedirs(output_dir, exist_ok=True)

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# ============ ASSETS ============
PIECE_SETS = ['alpha', 'cburnett', 'merida', 'california', 'cardinal', 'dubrovny', 'icpieces', 'maestro']
PIECE_CACHE = {}

def get_piece_image(piece_set, piece_name, tile_size):
    key = f"{piece_set}_{piece_name}_{tile_size}"
    if key not in PIECE_CACHE:
        path = os.path.join(base_dir, "piece_sets", piece_set, f"{piece_name}.png")
        if os.path.exists(path):
            img = Image.open(path).convert("RGBA")
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            PIECE_CACHE[key] = img
        else:
            return None
    return PIECE_CACHE[key]

def fen_to_grid(fen):
    rows = fen.split()[0].split('/')
    grid = []
    for row in rows:
        grid_row = []
        for char in row:
            if char.isdigit():
                grid_row.extend([None] * int(char))
            else:
                grid_row.append(char)
        grid.append(grid_row)
    return grid

def apply_jpeg_compression(img, quality=None):
    """Simulate JPEG compression artifacts by round-tripping through JPEG encoding"""
    quality = quality or random.randint(30, 95)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def augment_tile(tile_img):
    """Apply random real-world augmentations to a single tile PIL image"""
    # 1. Random slight crop then resize back (simulate tile misalignment)
    w, h = tile_img.size  # should be IMG_SIZE x IMG_SIZE  
    if random.random() < 0.6:
        # Crop 0-3px from random sides, simulating imperfect board splitting
        left   = random.randint(0, 3)
        top    = random.randint(0, 3)
        right  = w - random.randint(0, 3)
        bottom = h - random.randint(0, 3)
        if right > left + 8 and bottom > top + 8:
            tile_img = tile_img.crop((left, top, right, bottom))
            tile_img = tile_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # 2. JPEG compression (most important! bridges real vs synthetic gap)
    if random.random() < 0.75:
        tile_img = apply_jpeg_compression(tile_img, quality=random.randint(25, 90))
        tile_img = tile_img.convert("L")

    # 3. Random Gaussian blur (simulate screenshots scaled down)
    if random.random() < 0.5:
        radius = random.uniform(0.2, 1.2)
        tile_img = tile_img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 4. Brightness jitter (different monitor gammas)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.4)
        tile_img = ImageEnhance.Brightness(tile_img).enhance(factor)

    # 5. Contrast jitter
    if random.random() < 0.4:
        factor = random.uniform(0.7, 1.4)
        tile_img = ImageEnhance.Contrast(tile_img).enhance(factor)

    return tile_img

def render_board_to_tiles(fen, theme, piece_set, add_coords):
    """Renders a FEN to a list of 64 augmented numpy tiles using PIL"""
    # Random tile size so model sees different scales
    ts = random.choice([40, 50, 50, 50, 64])  # 50px most common

    # If coordinates are enabled, add a border around the board
    border = ts // 2 if add_coords else 0
    board_size = 8 * ts
    img_width = board_size + 2 * border
    img_height = board_size + 2 * border
    img = Image.new("RGB", (img_width, img_height), theme[0])  # background

    draw = ImageDraw.Draw(img)

    # Draw squares in the center area
    for row in range(8):
        for col in range(8):
            x0 = border + col * ts
            y0 = border + row * ts
            x1 = x0 + ts
            y1 = y0 + ts
            color = theme[0] if (row + col) % 2 == 0 else theme[1]
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Draw coordinates on the border if requested
    if add_coords:
        # Try to use a TrueType font; fallback to default if not available
        try:
            from PIL import ImageFont
            # You can replace 'arial.ttf' with any .ttf file you have
            font = ImageFont.truetype("arial.ttf", size=ts//3)
        except:
            font = ImageFont.load_default()

        # Files (a-h) on the bottom border
        files = ['a','b','c','d','e','f','g','h']
        for col, file in enumerate(files):
            x = border + col * ts + ts//2 - ts//6   # rough centering
            y = img_height - border//2 - ts//6
            draw.text((x, y), file, fill="black", font=font)

        # Ranks (1-8) on the left border (top to bottom)
        ranks = ['8','7','6','5','4','3','2','1']
        for row, rank in enumerate(ranks):
            x = border//2 - ts//6
            y = border + row * ts + ts//2 - ts//6
            draw.text((x, y), rank, fill="black", font=font)

        # Optionally add coordinates on the top and right borders if desired
        # (left as exercise; the code above already gives a realistic look)

    # Paste pieces
    piece_map = {'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
                 'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'}
    grid = fen_to_grid(fen)
    for row in range(8):
        for col in range(8):
            char = grid[row][col]
            if char in piece_map:
                p_img = get_piece_image(piece_set, piece_map[char], ts)
                if p_img:
                    img.paste(p_img, (border + col * ts, border + row * ts), p_img)

    # Board-level JPEG compression (most realistic — full board compressed then split)
    if random.random() < 0.7:
        img = apply_jpeg_compression(img, quality=random.randint(40, 90))

    # Convert to Grayscale
    full_img = img.convert("L")

    # Board-level blur (simulate screenshot at wrong resolution)
    if random.random() < 0.3:
        full_img = full_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # Tile extraction and labeling
    tiles = []
    labels = []

    # IMPORTANT: This import must be available in your environment.
    # If you don't have it, replace with a simple transform, e.g.:
    # from torchvision import transforms
    # transform = transforms.Compose([transforms.ToTensor()])
    from src.chessimg2pos.chessdataset import create_image_transforms
    transform = create_image_transforms(use_grayscale=True)

    for row in range(8):
        for col in range(8):
            left = border + col * ts
            top = border + row * ts
            tile = full_img.crop((left, top, left + ts, top + ts))
            tile = tile.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            # Apply per-tile augmentations (more aggressive than board-level)
            tile = augment_tile(tile)

            char = grid[row][col]
            if char is None:
                label = 0  # '1' = empty
            else:
                label = FEN_CHARS.index(char)

            tile_tensor = transform(tile)
            tiles.append(tile_tensor.numpy())
            labels.append(label)

    return tiles, labels

def generate_random_fen():
    """Generates a realistic chess position from legal moves"""
    board = chess.Board()  # Start from standard position

    # Make 5-80 random legal moves for varied positions
    num_moves = random.randint(5, 80)
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves or board.is_game_over():
            break
        board.push(random.choice(legal_moves))

    return board.fen().split()[0]

def process_chunk(name):
    print(f"🎨 Generating {name}...")
    start = time.time()

    all_imgs = []
    all_lbls = []

    for i in range(BOARDS_PER_CHUNK):
        fen = generate_random_fen()
        theme = random.choice(THEMES)
        piece_set = random.choice(PIECE_SETS)

        # Determine whether to add coordinates based on global mode
        if COORDS_MODE == 'always':
            use_coords = True
        elif COORDS_MODE == 'never':
            use_coords = False
        else:  # 'random'
            use_coords = random.choice([True, False])

        tiles, labels = render_board_to_tiles(fen, theme, piece_set, add_coords=use_coords)
        all_imgs.extend(tiles)
        all_lbls.extend(labels)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"   {name}: {i+1}/{BOARDS_PER_CHUNK} boards | {elapsed:.0f}s elapsed")

    x = torch.from_numpy(np.stack(all_imgs))  # (N, 1, 32, 32) float32
    y = torch.tensor(all_lbls, dtype=torch.uint8)

    path = os.path.join(output_dir, f"{name}.pt")
    torch.save({'x': x, 'y': y}, path)
    print(f"✅ Saved {path} | {len(all_imgs):,} tiles | {time.time()-start:.1f}s")

def main():
    print("🚀 LICHESS TENSOR GENERATOR v2 - REAL-WORLD SIM MODE")
    print(f"📊 Target: {(CHUNKS_TRAIN + CHUNKS_VAL) * BOARDS_PER_CHUNK:,} boards → {(CHUNKS_TRAIN + CHUNKS_VAL) * BOARDS_PER_CHUNK * 64:,} tiles")
    print(f"📂 Output: {output_dir}")
    print(f"🎨 Augmentations: JPEG compression, Gaussian blur, tile misalignment, brightness/contrast jitter")
    print(f"🏷️  Coordinate mode: {COORDS_MODE}")
    print()

    start_total = time.time()

    # Generate Validation first
    for i in range(CHUNKS_VAL):
        process_chunk(f"val_chunk_{i:02d}")

    # Generate Training
    for i in range(CHUNKS_TRAIN):
        process_chunk(f"train_chunk_{i:02d}")

    print("" + "=" * 60)
    print(f"🎉 SYNTHETIC DATA READY! Total Time: {format_time(time.time() - start_total)}")
    print(f"📁 Tensors saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()