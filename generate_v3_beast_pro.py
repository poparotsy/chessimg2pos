"""
🎨 V3 BEAST GENERATOR - PRO VERSION
Generates 64x64 RGB Chess Tiles with real-world noise (Highlights, Coords, Jitter).
Saves in NCHW byte format for maximum Kaggle performance.
"""

import os
import time
import random
import chess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ============ CONFIG ============
IMG_SIZE = 64
CHUNKS_TRAIN = 10
BOARDS_PER_CHUNK = 1000
FEN_CHARS = "1RNBQKPrnbqkp"

# Diverse themes and piece sets
THEMES = [
    ("#f0d9b5", "#b58863"), ("#ffffff", "#769656"), ("#dee3e6", "#8ca2ad"),
    ("#ffffff", "#999999"), ("#f0e9c5", "#c5a55a"), ("#faf0d7", "#8b6343")
]
HIGHLIGHTS = ["#f7f769", "#b9ca43", "#64b5f6", "#f44336", None]
PIECE_SETS = ['alpha', 'cburnett', 'merida', 'california', 'cardinal',
              'dubrovny', 'gioco', 'maestro']


def get_piece_image(p_set, p_name, size):
    """Loads and resizes a chess piece image."""
    path = os.path.join("piece_sets", p_set, f"{p_name}.png")
    if os.path.exists(path):
        return Image.open(path).convert(
            "RGBA").resize((size, size), Image.LANCZOS)
    return None


def apply_jitter(img):
    """Applies random brightness and contrast jitter."""
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    return img


def render_board_v3(fen):
    """Renders a full board and returns 64 extracted tiles."""
    theme = random.choice(THEMES)
    ts = random.choice([64, 72, 80])
    img = Image.new("RGB", (8 * ts, 8 * ts), theme[0])
    draw = ImageDraw.Draw(img)

    # 1. Highlights and Grid
    hl_sqs = random.sample(range(64), 2) if random.random() > 0.4 else []
    hl_color = random.choice(HIGHLIGHTS)
    for r in range(8):
        for c in range(8):
            color = theme[0] if (r + c) % 2 == 0 else theme[1]
            if (r * 8 + c) in hl_sqs and hl_color:
                color = hl_color
            draw.rectangle([c * ts, r * ts, (c + 1) *
                           ts, (r + 1) * ts], fill=color)
            if random.random() > 0.9:  # Internal coordinates
                draw.text(
                    (c * ts + 2,
                     r * ts + 2),
                    random.choice("abcdefgh12345678"),
                    fill="#444")

    # 2. Pieces
    piece_map = {'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
                 'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'}
    grid = []
    rows = fen.split()[0].split('/')
    for row in rows:
        r_list = []
        for char in row:
            if char.isdigit():
                r_list.extend([None] * int(char))
            else:
                r_list.append(char)
        grid.append(r_list)

    p_set = random.choice(PIECE_SETS)
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char:
                p_img = get_piece_image(p_set, piece_map[char], ts)
                if p_img:
                    off = (random.randint(-2, 2), random.randint(-2, 2))
                    img.paste(p_img, (c * ts + off[0], r * ts + off[1]), p_img)

    # 3. Global Noise
    img = apply_jitter(img)
    if random.random() > 0.5:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(
                    0.1, 0.6)))

    # 4. Extraction
    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = img.crop((c * ts, r * ts, (c + 1) * ts,
                            (r + 1) * ts)).resize((IMG_SIZE, IMG_SIZE))
            # Save as CHW uint8
            tiles.append(
                np.transpose(
                    np.array(
                        tile, dtype=np.uint8), (2, 0, 1)))
            labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)

    return np.stack(tiles), np.array(labels, dtype=np.uint8)


def main():
    """Main generation loop."""
    print("🚀 Starting V3 Pro Generator...")
    os.makedirs("tensors_v3", exist_ok=True)
    for i in range(CHUNKS_TRAIN):
        start = time.time()
        all_x, all_y = [], []
        for _ in range(BOARDS_PER_CHUNK):
            board = chess.Board()
            for _ in range(random.randint(5, 60)):
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
            x_board, y_board = render_board_v3(board.fen().split()[0])
            all_x.append(x_board)
            all_y.append(y_board)

        torch.save({'x': torch.from_numpy(np.concatenate(all_x)),
                    'y': torch.from_numpy(np.concatenate(all_y))},
                   f"tensors_v3/train_chunk_{i}.pt")
        print(f"✅ Chunk {i} | Time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
