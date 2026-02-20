#!/usr/bin/env python3
import sys
import json
import re
from chessimg2pos import ChessPositionPredictor

ultra_model_path = "./models/model_tensor_beast.pt"

def compress_fen(board_fen):
    """Convert 11111 notation to standard FEN (5)"""
    rows = board_fen.split('/')
    compressed_rows = []
    for row in rows:
        # Replace consecutive 1s with their count
        compressed = re.sub(r'1+', lambda m: str(len(m.group())), row)
        compressed_rows.append(compressed)
    return '/'.join(compressed_rows)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python3 chess_recognizer_tensor_model.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Use ultra model for better accuracy
        #predictor = ChessPositionPredictor("./models/model_ultra_fresh.pt", classifier="ultra")
        predictor = ChessPositionPredictor(ultra_model_path, classifier="ultra")
        result = predictor.predict_chessboard(image_path, return_tiles=False)
        
        # Calculate average confidence per square (more meaningful)
        avg_confidence = sum(p[2] for p in result["predictions"]) / 64
        min_confidence = min(p[2] for p in result["predictions"])
        
        # Compress FEN notation (11111 -> 5)
        board_position = compress_fen(result["fen"])
        complete_fen = f"{board_position} w - - 0 1"
        
        output = {
            "success": True,
            "fen": complete_fen,
            "board": board_position,
            "avg_confidence": round(avg_confidence, 4),
            "min_confidence": round(min_confidence, 4),
            "overall_confidence": result["confidence"]
        }
    except Exception as e:
        output = {
            "success": False,
            "error": str(e)
        }
    
    print(json.dumps(output))

if __name__ == "__main__":
    main()
