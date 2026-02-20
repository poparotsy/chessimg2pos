import os
import cairosvg

def main():
    sets = ['alpha', 'cburnett', 'merida', 'california', 'cardinal', 'dubrovny', 'icpieces', 'maestro']
    pieces = ['wK', 'wQ', 'wR', 'wB', 'wN', 'wP', 'bK', 'bQ', 'bR', 'bB', 'bN', 'bP']
    
    out_dir = os.path.join(os.path.dirname(__file__), "piece_sets")
    os.makedirs(out_dir, exist_ok=True)
    
    for s in sets:
        s_dir = os.path.join(out_dir, s)
        os.makedirs(s_dir, exist_ok=True)
        print(f"Downloading {s}...")
        for p in pieces:
            out_file = os.path.join(s_dir, f"{p}.png")
            if os.path.exists(out_file):
                continue
            
            url = f"https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/{s}/{p}.svg"
            try:
                cairosvg.svg2png(url=url, write_to=out_file, output_width=50, output_height=50)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")

if __name__ == "__main__":
    main()
