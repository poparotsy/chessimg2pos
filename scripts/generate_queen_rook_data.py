#!/usr/bin/env python3
"""
Generate focused training boards with queens and rooks in various positions
to help the model distinguish between them better
"""
import os
import random
from PIL import Image, ImageDraw

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "images", "chessboards", "queen_rook_focus")
os.makedirs(output_dir, exist_ok=True)

def generate_queen_rook_boards(num_boards=1000):
    """
    Generate boards with various combinations of queens and rooks
    to help the model learn to distinguish them
    """
    pieces = ['R', 'Q', 'r', 'q', 'K', 'k', 'P', 'p', 'N', 'n', 'B', 'b']
    
    boards_generated = 0
    
    for i in range(num_boards):
        # Create a board with emphasis on queens and rooks
        board = [['1' for _ in range(8)] for _ in range(8)]
        
        # Always place at least one queen and one rook
        # White pieces
        q_row, q_col = random.randint(0, 7), random.randint(0, 7)
        board[q_row][q_col] = 'Q'
        
        r_row, r_col = random.randint(0, 7), random.randint(0, 7)
        while (r_row, r_col) == (q_row, q_col):
            r_row, r_col = random.randint(0, 7), random.randint(0, 7)
        board[r_row][r_col] = 'R'
        
        # Black pieces
        bq_row, bq_col = random.randint(0, 7), random.randint(0, 7)
        while board[bq_row][bq_col] != '1':
            bq_row, bq_col = random.randint(0, 7), random.randint(0, 7)
        board[bq_row][bq_col] = 'q'
        
        br_row, br_col = random.randint(0, 7), random.randint(0, 7)
        while board[br_row][br_col] != '1':
            br_row, br_col = random.randint(0, 7), random.randint(0, 7)
        board[br_row][br_col] = 'r'
        
        # Add kings (required)
        wk_row, wk_col = random.randint(0, 7), random.randint(0, 7)
        while board[wk_row][wk_col] != '1':
            wk_row, wk_col = random.randint(0, 7), random.randint(0, 7)
        board[wk_row][wk_col] = 'K'
        
        bk_row, bk_col = random.randint(0, 7), random.randint(0, 7)
        while board[bk_row][bk_col] != '1':
            bk_row, bk_col = random.randint(0, 7), random.randint(0, 7)
        board[bk_row][bk_col] = 'k'
        
        # Randomly add a few more pieces
        for _ in range(random.randint(2, 6)):
            piece = random.choice(pieces)
            row, col = random.randint(0, 7), random.randint(0, 7)
            if board[row][col] == '1':
                board[row][col] = piece
        
        # Convert to filename format
        filename = '-'.join([''.join(row) for row in board]) + '.png'
        filepath = os.path.join(output_dir, filename)
        
        # Note: This creates placeholder files - you'd need actual chess board images
        # For now, just create the filename structure
        print(f"Would generate: {filename}")
        boards_generated += 1
        
        if boards_generated >= 10:  # Just show first 10
            break
    
    return boards_generated

print("📊 This script shows the concept of generating focused training data")
print("🎯 Focus: Queen vs Rook distinction")
print("\nTo actually generate images, you need:")
print("1. A chess board renderer (like python-chess + cairosvg)")
print("2. Or use existing chess diagram generators")
print("\n💡 Better approach: Find existing datasets online")
print("   - Lichess puzzle database")
print("   - Chess.com positions")
print("   - SCID database screenshots")

generate_queen_rook_boards(10)
