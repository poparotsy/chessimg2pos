#!/usr/bin/env python3
"""
Train Ultra model using the pre-generated tiles in images/tiles_all
"""
import os
from chessimg2pos import ChessRecognitionTrainer

base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_all_dir = os.path.join(base_dir, "images", "tiles_all")
tiles_dir = os.path.join(base_dir, "images", "tiles")
model_path = os.path.join(base_dir, "models", "model_ultra_all.pt")

# Temporarily rename tiles_all to tiles
print(f"📁 Renaming tiles_all -> tiles temporarily...")
os.rename(tiles_all_dir, tiles_dir)

try:
    print(f"💾 Model will be saved to: {model_path}")
    
    # Pass a dummy images_dir - the trainer will calculate tiles_dir as dirname(images_dir) + "/tiles"
    # So if we pass "images/dummy", it becomes "images/tiles" which is what we want
    dummy_images_dir = os.path.join(base_dir, "images", "dummy")
    
    trainer = ChessRecognitionTrainer(
        images_dir=dummy_images_dir,
        model_path=model_path,
        generate_tiles=False,
        epochs=15,
        overwrite=True
    )
    
    print("\n🚀 Starting training with Ultra classifier...")
    model, device, accuracy = trainer.train(classifier="ultra")
    
    print(f"\n✅ Training complete!")
    print(f"📈 Final accuracy: {accuracy:.2%}")
    print(f"💾 Model saved to: {model_path}")
finally:
    # Rename back
    print(f"\n📁 Renaming tiles -> tiles_all...")
    if os.path.exists(tiles_dir):
        os.rename(tiles_dir, tiles_all_dir)

print(f"\n🔧 Update chess_recognizer_ultra.py to use: ./models/model_ultra_all.pt")
