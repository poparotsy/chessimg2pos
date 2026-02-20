#!/usr/bin/env python3
import sys
import os
import glob
import re

# Import the predictor class directly
sys.path.insert(0, 'src')
from chessimg2pos.predictor import ChessPositionPredictor

def compress_fen(board_fen):
    """Convert 11111 notation to standard FEN (5)"""
    rows = board_fen.split('/')
    compressed_rows = []
    for row in rows:
        compressed = re.sub(r'1+', lambda m: str(len(m.group())), row)
        compressed_rows.append(compressed)
    return '/'.join(compressed_rows)

def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else 'images_4_test'
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(test_dir, '*.*')))
    if not image_paths:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(image_paths)} images in {test_dir}. Validating...")
    
    model_path = './models/model_tensor_beast.pt'
    print(f"Using model: {model_path}\n")
    predictor = ChessPositionPredictor(model_path, classifier='ultra', use_grayscale=True)
    
    all_confs = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            result = predictor.predict_chessboard(img_path)
            formatted_fen = result['fen']
            
            # Use mean per-tile confidence (product of 64 is always ~0.0)
            predictions = result['predictions']
            mean_conf = sum(p[2] for p in predictions) / len(predictions)
            min_conf  = min(p[2] for p in predictions)
            all_confs.append(mean_conf)
            
            board_position = compress_fen(formatted_fen)
            complete_fen = f"{board_position} w - - 0 1"
            
            base, ext = os.path.splitext(filename)
            safe_fen = board_position.replace('/', '-')
            output_file = os.path.join(output_dir, f"{base}_{safe_fen}{ext}")
            
            with open(img_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
                    
            print(f"✅ {filename}")
            print(f"   FEN:      {board_position}")
            print(f"   Avg conf: {mean_conf:.3f} | Min conf: {min_conf:.3f}\n")
            
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}\n")

    if all_confs:
        print(f"{'='*50}")
        print(f"📊 Summary: {len(all_confs)}/{len(image_paths)} images processed")
        print(f"   Mean avg confidence: {sum(all_confs)/len(all_confs):.3f}")

if __name__ == "__main__":
    main()
