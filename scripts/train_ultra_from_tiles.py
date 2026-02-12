#!/usr/bin/env python3
"""
Use the existing tiles that were already generated during enhanced model training
to train an ultra model - much faster and avoids filename parsing issues
"""
import os
import glob
from chessimg2pos import ChessRecognitionTrainer

base_dir = os.path.dirname(os.path.abspath(__file__))

# Check if tiles already exist from previous training
tiles_dir = os.path.join(base_dir, "images", "tiles")
if os.path.exists(tiles_dir):
    tile_count = len(glob.glob(os.path.join(tiles_dir, "*", "*.png")))
    print(f"✅ Found existing tiles directory with {tile_count} tiles")
    print(f"📁 Using: {tiles_dir}")
else:
    print("❌ No tiles directory found. The enhanced model should have created one.")
    print("   Run the enhanced model training first, or use chess_recognizer.py (enhanced)")
    exit(1)

# Train ultra model using existing tiles
model_path = os.path.join(base_dir, "models", "model_ultra_retrained.pt")

print("\n🚀 Training Ultra model with existing tile data...")
trainer = ChessRecognitionTrainer(
    images_dir=tiles_dir,  # Point to tiles, not boards
    model_path=model_path,
    generate_tiles=False,  # Tiles already exist!
    epochs=15,
    overwrite=True
)

model, device, accuracy = trainer.train(classifier="ultra")

print(f"\n✅ Training complete!")
print(f"📈 Final accuracy: {accuracy:.2%}")
print(f"💾 Model saved to: {model_path}")
