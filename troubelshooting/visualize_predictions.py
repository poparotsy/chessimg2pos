#!/usr/bin/env python3
"""
Visualize model predictions tile by tile
"""
import sys
sys.path.insert(0, 'src')
from chessimg2pos.predictor import ChessPositionPredictor
from chessimg2pos.chessboard_image import get_chessboard_tiles
from PIL import Image
import matplotlib.pyplot as plt

image_path = sys.argv[1] if len(sys.argv) > 1 else 'this.png'

# Load model
predictor = ChessPositionPredictor('./models/model_tensor_beast.pt', classifier='ultra', use_grayscale=True)

# Get tiles
tiles = get_chessboard_tiles(image_path, use_grayscale=True)

# Predict and show first 16 tiles (2 rows)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle(f'Model Predictions on {image_path}')

for i in range(16):
    row = i // 8
    col = i % 8
    
    tile = tiles[i]
    fen_char, confidence = predictor.predict_tile(tile)
    
    axes[row, col].imshow(tile, cmap='gray')
    axes[row, col].set_title(f'{fen_char}\n{confidence:.2f}', fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('tile_predictions.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization to tile_predictions.png")
print("\nFirst 16 tiles (rows 8 and 7):")
for i in range(16):
    fen_char, confidence = predictor.predict_tile(tiles[i])
    square = chr(97 + i%8) + str(8 - i//8)
    print(f"{square}: '{fen_char}' (confidence: {confidence:.2%})")
