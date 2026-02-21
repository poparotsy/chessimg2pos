import os, io, time, torch, random, chess, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torchvision import transforms

# ============ CONFIG ============
IMG_SIZE = 64
CHUNKS_TRAIN = 10 
BOARDS_PER_CHUNK = 1000 
FEN_CHARS = "1RNBQKPrnbqkp"

# Added Grayscale themes for "Newspaper" robustness
THEMES = [
    ("#f0d9b5", "#b58863"), # Lichess Wood
    ("#ffffff", "#769656"), # Lichess Green
    ("#dee3e6", "#8ca2ad"), # Marble
    ("#ffffff", "#999999"), # Greyscale (Newspaper)
    ("#f0e9c5", "#c5a55a")  # Chess.com Brown
]
HIGHLIGHTS = ["#f7f769", "#b9ca43", "#64b5f6", None] # Yellow, Green, Blue, or None

PIECE_SETS = ['alpha', 'cburnett', 'merida']
base_dir = os.getcwd()

def get_piece_image(piece_set, piece_name, size):
    path = os.path.join(base_dir, "piece_sets", piece_set, f"{piece_name}.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    return None

def render_board_v3(fen):
    theme = random.choice(THEMES)
    ts = random.choice([64, 72, 80]) # Varied internal resolution
    img = Image.new("RGB", (8*ts, 8*ts), theme[0])
    draw = ImageDraw.Draw(img)
    
    # 1. Random Move Highlights (simulate real gameplay)
    hl_sqs = random.sample(range(64), 2) if random.random() > 0.4 else []
    hl_color = random.choice(HIGHLIGHTS)

    # 2. Draw Squares & Highlights
    for r in range(8):
        for c in range(8):
            color = theme[0] if (r+c)%2==0 else theme[1]
            if (r*8 + c) in hl_sqs and hl_color: color = hl_color
            draw.rectangle([c*ts, r*ts, (c+1)*ts, (r+1)*ts], fill=color)
            
            # 3. Internal Coordinates (The Twitter Fix)
            if random.random() > 0.85:
                draw.text((c*ts+2, r*ts+2), random.choice("abcdefgh12345678"), fill="#444")

    # 4. Paste Pieces
    piece_map = {'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
                 'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'}
    grid = []
    rows = fen.split()[0].split('/')
    for row in rows:
        r_list = []
        for char in row:
            if char.isdigit(): r_list.extend([None] * int(char))
            else: r_list.append(char)
        grid.append(r_list)

    active_set = random.choice(PIECE_SETS)
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char:
                p_img = get_piece_image(active_set, piece_map[char], ts)
                if p_img:
                    # Slight random jitter in piece placement for 3D simulation
                    off_x, off_y = random.randint(-2,2), random.randint(-2,2)
                    img.paste(p_img, (c*ts + off_x, r*ts + off_y), p_img)

    # 5. Distortions
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.6)))
    
    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = img.crop((c*ts, r*ts, (c+1)*ts, (r+1)*ts)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            # STORAGE OPTIMIZATION: Save as bytes (uint8)
            tiles.append(np.array(tile, dtype=np.uint8))
            labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)
            
    return np.stack(tiles), np.array(labels, dtype=np.uint8)

def main():
    print(f"🚀 Starting V3 Generator (RGB {IMG_SIZE}px)...")
    os.makedirs("tensors_v3", exist_ok=True)
    
    for i in range(CHUNKS_TRAIN):
        start_time = time.time()
        all_x, all_y = [], []
        
        for _ in range(BOARDS_PER_CHUNK):
            board = chess.Board()
            for _ in range(random.randint(5, 60)):
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
            
            x, y = render_board_v3(board.fen().split()[0])
            all_x.append(x)
            all_y.append(y)
        
        # Save as uint8 to save 75% disk space
        x_tensor = torch.from_numpy(np.concatenate(all_x))
        y_tensor = torch.from_numpy(np.concatenate(all_y))
        
        torch.save({'x': x_tensor, 'y': y_tensor}, f"tensors_v3/train_chunk_{i}.pt")
        print(f"✅ Chunk {i} saved | Time: {time.time()-start_time:.1f}s | Size: {x_tensor.element_size() * x_tensor.nelement() / 1e6:.1f} MB")

if __name__ == "__main__":
    main()

