#!/usr/bin/env python3
"""
Train an Ultra model for better chess piece recognition
This will use your existing training images in images/chessboards/
"""
import os
import glob
from chessimg2pos import ChessRecognitionTrainer

# Use absolute path to avoid confusion
base_dir = os.path.dirname(os.path.abspath(__file__))

# The trainer expects images directly in the directory, not subdirectories
# So we'll point it to one of the subdirectories with the most images
images_dir = os.path.join(base_dir, "images", "chessboards", "generated")
model_path = os.path.join(base_dir, "models", "model_ultra.pt")

# Check how many images we have
num_images = len(glob.glob(os.path.join(images_dir, "*.png")))
print(f"📁 Images directory: {images_dir}")
print(f"📊 Found {num_images} training images")
print(f"💾 Model will be saved to: {model_path}")

if num_images == 0:
    print("❌ No images found! Check the path.")
    exit(1)

# Train with Ultra classifier for best accuracy
trainer = ChessRecognitionTrainer(
    images_dir=images_dir,
    model_path=model_path,
    generate_tiles=True,  # Generate tiles from board images
    epochs=15,  # More epochs = better accuracy
    overwrite=True
)

print("\n🚀 Starting training with Ultra classifier...")
print("📊 This will take some time depending on your dataset size")
model, device, accuracy = trainer.train(classifier="ultra")

print(f"\n✅ Training complete!")
print(f"📈 Final accuracy: {accuracy:.2%}")
print(f"💾 Model saved to: {model_path}")
print(f"\n🔧 Update chess_recognizer.py to use this model")
