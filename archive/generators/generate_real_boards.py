#!/usr/bin/env python3
"""
Generate real chess board images using python-chess library
Focused on queen/rook distinction
"""

# First install: pip install python-chess cairosvg pillow

import chess
import chess.svg
import random
import os
from io import BytesIO

try:
    import cairosvg
    from PIL import Image
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install python-chess cairosvg pillow")
    exit(1)

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "images", "chessboards", "generated_qr")
os.makedirs(output_dir, exist_ok=True)

def fen_to_filename(fen):
    """Convert FEN to filename format"""
    board_part = fen.split()[0]
    return board_part.replace('/', '-') + '.png'

def generate_board_image(fen, output_path, size=400):
    """Generate a chess board image from FEN"""
    board = chess.Board(fen)
    svg_data = chess.svg.board(board, size=size)
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    
    # Save as PNG
    img = Image.open(BytesIO(png_data))
    img.save(output_path)

def generate_queen_rook_positions(count=500):
    """Generate positions emphasizing queens and rooks"""
    
    for i in range(count):
        board = chess.Board(None)  # Empty board
        
        # Place kings (required)
        board.set_piece_at(random.choice(list(chess.SQUARES)), chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(random.choice([sq for sq in chess.SQUARES if board.piece_at(sq) is None]), 
                          chess.Piece(chess.KING, chess.BLACK))
        
        # Place queens and rooks (emphasis)
        for _ in range(random.randint(1, 2)):  # 1-2 white queens
            empty_sq = random.choice([sq for sq in chess.SQUARES if board.piece_at(sq) is None])
            board.set_piece_at(empty_sq, chess.Piece(chess.QUEEN, chess.WHITE))
        
        for _ in range(random.randint(1, 2)):  # 1-2 white rooks
            empty_sq = random.choice([sq for sq in chess.SQUARES if board.piece_at(sq) is None])
            board.set_piece_at(empty_sq, chess.Piece(chess.ROOK, chess.WHITE))
        
        for _ in range(random.randint(1, 2)):  # 1-2 black queens
            empty_sq = random.choice([sq for sq in chess.SQUARES if board.piece_at(sq) is None])
            board.set_piece_at(empty_sq, chess.Piece(chess.QUEEN, chess.BLACK))
        
        for _ in range(random.randint(1, 2)):  # 1-2 black rooks
            empty_sq = random.choice([sq for sq in chess.SQUARES if board.piece_at(sq) is None])
            board.set_piece_at(empty_sq, chess.Piece(chess.ROOK, chess.BLACK))
        
        # Add some other pieces randomly
        for _ in range(random.randint(2, 6)):
            piece_type = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP])
            color = random.choice([chess.WHITE, chess.BLACK])
            empty_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
            if empty_squares:
                board.set_piece_at(random.choice(empty_squares), chess.Piece(piece_type, color))
        
        # Generate image
        fen = board.fen()
        filename = fen_to_filename(fen)
        filepath = os.path.join(output_dir, filename)
        
        try:
            generate_board_image(fen, filepath)
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{count} boards...")
        except Exception as e:
            print(f"Failed to generate board {i}: {e}")
    
    print(f"\n✅ Generated {count} boards in {output_dir}")
    print(f"🚀 Now run: python3 train_with_existing_tiles.py")

if __name__ == "__main__":
    print("🎨 Generating chess boards with emphasis on Queens and Rooks...")
    generate_queen_rook_positions(500)
