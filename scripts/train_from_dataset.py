#!/usr/bin/env python3
"""
Train model using tiles from the downloaded 3D chess dataset
"""
import os
from chessimg2pos import ChessRecognitionTrainer

base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dataset_dir = os.path.join(base_dir, "images", "tiles_dataset")
tiles_dir = os.path.join(base_dir, "images", "tiles")
model_path = os.path.join(base_dir, "models", "model_3d_dataset.pt")

# Temporarily rename for trainer
print(f"📁 Using tiles from: {tiles_dataset_dir}")
os.rename(tiles_dataset_dir, tiles_dir)

try:
    trainer = ChessRecognitionTrainer(
        images_dir=os.path.join(base_dir, "images", "dummy"),
        model_path=model_path,
        generate_tiles=False,
        epochs=15,
        overwrite=True
    )
    
    print("\n🚀 Training with 3D dataset tiles...")
    model, device, accuracy = trainer.train(classifier="ultra")
    
    print(f"\n✅ Training complete!")
    print(f"📈 Final accuracy: {accuracy:.2%}")
    print(f"💾 Model: {model_path}")
finally:
    if os.path.exists(tiles_dir):
        os.rename(tiles_dir, tiles_dataset_dir)
