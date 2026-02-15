#!/usr/bin/env python3
import sys
import json
import re
from chessimg2pos import predict_fen

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
        print(json.dumps({"error": "Usage: chess_recognizer.py <image_path>"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        result = predict_fen(image_path, output_type="full")
        # Compress FEN notation (11111 -> 5)
        board_position = compress_fen(result["fen"])
        complete_fen = f"{board_position} w - - 0 1"
        
        output = {
            "success": True,
            "fen": complete_fen,
            "board": board_position,
            "confidence": result["confidence"]
        }
    except Exception as e:
        output = {
            "success": False,
            "error": str(e)
        }
    
    print(json.dumps(output))

if __name__ == "__main__":
    main()
