#!/usr/bin/env python3
"""
Quick script to generate and save a synthetic chessboard
using your v2 generator logic, so you can compare with real screenshots.
"""

import os
import io
import random
import chess
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
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

PIECE_SETS = ['alpha', 'cburnett', 'merida', 'california', 'cardinal', 'dubrovny', 'icpieces', 'maestro']
PIECE_CACHE = {}

def get_piece_image(piece_set, piece_name, tile_size):
    key = f"{piece_set}_{piece_name}_{tile_size}"
    if key not in PIECE_CACHE:
        # Adjust this path to where your piece_sets are located
        path = os.path.join("piece_sets", piece_set, f"{piece_name}.png")
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
    quality = quality or random.randint(30, 95)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def render_full_board(fen, theme, piece_set, add_coords=True, tile_size=50):
    """Render a full chessboard (not tiles) for visualization"""
    border = tile_size // 2 if add_coords else 0
    board_size = 8 * tile_size
    img_width = board_size + 2 * border
    img_height = board_size + 2 * border
    
    img = Image.new("RGB", (img_width, img_height), theme[0])
    draw = ImageDraw.Draw(img)

    # Draw squares
    for row in range(8):
        for col in range(8):
            x0 = border + col * tile_size
            y0 = border + row * tile_size
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            color = theme[0] if (row + col) % 2 == 0 else theme[1]
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Draw coordinates
    if add_coords:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", size=tile_size//3)
        except:
            font = ImageFont.load_default()

        files = ['a','b','c','d','e','f','g','h']
        for col, file in enumerate(files):
            x = border + col * tile_size + tile_size//2 - tile_size//6
            y = img_height - border//2 - tile_size//6
            draw.text((x, y), file, fill="black", font=font)

        ranks = ['8','7','6','5','4','3','2','1']
        for row, rank in enumerate(ranks):
            x = border//2 - tile_size//6
            y = border + row * tile_size + tile_size//2 - tile_size//6
            draw.text((x, y), rank, fill="black", font=font)

    # Paste pieces
    piece_map = {'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
                 'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'}
    grid = fen_to_grid(fen)
    for row in range(8):
        for col in range(8):
            char = grid[row][col]
            if char in piece_map:
                p_img = get_piece_image(piece_set, piece_map[char], tile_size)
                if p_img:
                    img.paste(p_img, (border + col * tile_size, border + row * tile_size), p_img)

    # Apply JPEG compression (your augmentation)
    if random.random() < 0.7:
        img = apply_jpeg_compression(img, quality=random.randint(40, 90))

    return img.convert("L")  # Convert to grayscale

def generate_random_fen():
    board = chess.Board()
    num_moves = random.randint(5, 80)
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves or board.is_game_over():
            break
        board.push(random.choice(legal_moves))
    return board.fen().split()[0]

def main():
    print("🎨 Generating sample synthetic chessboards...")
    
    # Generate 3 different boards with different themes
    for i in range(3):
        fen = generate_random_fen()
        theme = random.choice(THEMES)
        piece_set = random.choice(PIECE_SETS)
        
        print(f"\nBoard {i+1}:")
        print(f"  FEN: {fen}")
        print(f"  Theme: {theme}")
        print(f"  Piece set: {piece_set}")
        
        # Render with coordinates
        img_with_coords = render_full_board(fen, theme, piece_set, add_coords=True)
        img_with_coords.save(f"synthetic_board_{i+1}_with_coords.png")
        
        # Render without coordinates (v1 style)
        img_without_coords = render_full_board(fen, theme, piece_set, add_coords=False)
        img_without_coords.save(f"synthetic_board_{i+1}_without_coords.png")
        
        print(f"  ✅ Saved: synthetic_board_{i+1}_with_coords.png")
        print(f"  ✅ Saved: synthetic_board_{i+1}_without_coords.png")
    
    print("\n" + "="*60)
    print("✅ Done! Now go to lichess.org, take a screenshot,")
    print("   and compare the images side by side.")
    print("="*60)

if __name__ == "__main__":
    main()