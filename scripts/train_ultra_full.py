#!/usr/bin/env python3
"""
Prepare all training images by creating symlinks in a flat directory
Then train the ultra model with ALL available data
"""
import os
import glob
from pathlib import Path

base_dir = os.path.dirname(os.path.abspath(__file__))
chessboards_dir = os.path.join(base_dir, "images", "chessboards")
flat_dir = os.path.join(base_dir, "images", "chessboards_flat")

# Create flat directory with all images
print("📁 Creating flat directory with all training images...")
os.makedirs(flat_dir, exist_ok=True)

# Find all PNG files in subdirectories
all_images = glob.glob(os.path.join(chessboards_dir, "**", "*.png"), recursive=True)
print(f"📊 Found {len(all_images)} total images")

# Create symlinks (or copy if symlinks don't work)
linked = 0
for img_path in all_images:
    img_name = os.path.basename(img_path)
    link_path = os.path.join(flat_dir, img_name)
    
    if not os.path.exists(link_path):
        try:
            os.symlink(img_path, link_path)
            linked += 1
        except OSError:
            # Symlinks might not work on some systems, copy instead
            import shutil
            shutil.copy2(img_path, link_path)
            linked += 1

print(f"✅ Prepared {linked} images in {flat_dir}")

# Now train with all data
print("\n🚀 Starting training with Ultra classifier on ALL data...")
from chessimg2pos import ChessRecognitionTrainer

model_path = os.path.join(base_dir, "models", "model_ultra_full.pt")

trainer = ChessRecognitionTrainer(
    images_dir=flat_dir,
    model_path=model_path,
    generate_tiles=True,
    epochs=15,
    overwrite=True
)

model, device, accuracy = trainer.train(classifier="ultra")

print(f"\n✅ Training complete!")
print(f"📈 Final accuracy: {accuracy:.2%}")
print(f"💾 Model saved to: {model_path}")
