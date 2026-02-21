import os, io, time, torch, random, chess, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
from torchvision import transforms

# CONFIG
IMG_SIZE = 64 # Upgraded
CHUNKS_TRAIN = 10
BOARDS_PER_CHUNK = 1000 
FEN_CHARS = "1RNBQKPrnbqkp"

# Lichess/Chess.com style themes
THEMES = [("#f0d9b5", "#b58863"), ("#ffffff", "#769656"), ("#dee3e6", "#8ca2ad"), ("#f0e9c5", "#c5a55a"), ("#ffffff", "#999999"), ("#cccccc", "#777777"),]
HIGHLIGHTS = ["#f7f769", "#b9ca43", "#64b5f6"] # Yellow/Green/Blue move highlights

def get_piece_image(piece_set, piece_name, size):
    # Path logic: Ensure your piece_sets folder is reachable
    path = os.path.join("piece_sets", piece_set, f"{piece_name}.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    return None

def render_board_v3(fen):
    theme = random.choice(THEMES)
    ts = random.choice([64, 72, 80]) # Varied internal resolution
    img = Image.new("RGB", (8*ts, 8*ts), theme[0])
    draw = ImageDraw.Draw(img)
    
    # 1. Simulate Move Highlights (CRITICAL for Lichess screenshots)
    hl_sqs = random.sample(range(64), 2) if random.random() > 0.4 else []
    hl_color = random.choice(HIGHLIGHTS)

    # 2. Draw Squares & Highlights
    for r in range(8):
        for c in range(8):
            color = theme[0] if (r+c)%2==0 else theme[1]
            if (r*8 + c) in hl_sqs: color = hl_color
            draw.rectangle([c*ts, r*ts, (c+1)*ts, (r+1)*ts], fill=color)
            
            # 3. Random Internal Coordinates (The Twitter Fix)
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

    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char:
                p_img = get_piece_image(random.choice(['alpha', 'cburnett', 'merida']), piece_map[char], ts)
                if p_img:
                    # Slight random jitter in piece placement
                    img.paste(p_img, (c*ts + random.randint(-2,2), r*ts + random.randint(-2,2)), p_img)

    # 5. Global Blur/JPEG Noise
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.7)))
    
    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = img.crop((c*ts, r*ts, (c+1)*ts, (r+1)*ts)).resize((64, 64), Image.LANCZOS)
            tiles.append(transforms.ToTensor()(tile).numpy())
            labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)
            
    return np.stack(tiles), np.array(labels)

def main():
    os.makedirs("tensors_v3", exist_ok=True)
    for i in range(CHUNKS_TRAIN):
        all_x, all_y = [], []
        for _ in range(BOARDS_PER_CHUNK):
            board = chess.Board() # Generate random position
            for _ in range(random.randint(5, 50)):
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
            x, y = render_board_v3(board.fen().split()[0])
            all_x.append(x); all_y.append(y)
        
        torch.save({'x': torch.from_numpy(np.concatenate(all_x)), 
                    'y': torch.from_numpy(np.concatenate(all_y))}, 
                    f"tensors_v3/train_{i}.pt")
        print(f"✅ Created chunk {i}")

if __name__ == "__main__":
    main()

