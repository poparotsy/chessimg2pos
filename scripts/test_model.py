#!/usr/bin/env python3
"""Test model_enhanced.pt on sample images"""
from chessimg2pos import ChessPositionPredictor
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "model_enhanced.pt")

print(f"📦 Loading model: {model_path}")
predictor = ChessPositionPredictor(model_path)

# Test on sample images
test_images = [
    "images/chess_image.png",
]

# Add any other test images you have
import glob
test_images.extend(glob.glob("images/*.png")[:5])

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\n📷 Testing: {img_path}")
        try:
            result = predictor.predict_chessboard(img_path, return_tiles=True)
            print(f"   FEN: {result['fen']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
