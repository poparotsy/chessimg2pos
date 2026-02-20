import torch
import sys
sys.path.insert(0, 'src')
from chessimg2pos.chessboard_image import get_chessboard_tiles
from chessimg2pos.chessdataset import create_image_transforms
from PIL import Image

def main():
    print("--- INFERENCE TENSOR ---")
    image_path = "images_4_test/puzzle-00000.png"
    # Get tiles from image the same way predictor.py does
    tiles = get_chessboard_tiles(image_path, use_grayscale=True)
    tile_img = tiles[0]
    
    # Predictor does this:
    tile_img_copy = tile_img.copy()
    tile_img_copy = tile_img_copy.convert("L")
    transform = create_image_transforms(use_grayscale=True)
    inference_tensor = transform(tile_img_copy)
    
    print(f"Shape: {inference_tensor.shape}")
    print(f"Min: {inference_tensor.min().item():.4f}")
    print(f"Max: {inference_tensor.max().item():.4f}")
    print(f"Mean: {inference_tensor.mean().item():.4f}")
    print(f"Std: {inference_tensor.std().item():.4f}")

    print("\n--- TRAINING TENSOR ---")
    print("\n--- TRAINING TENSORS (First 10) ---")
    data = torch.load('tensor_dataset_synthetic/val_chunk_00.pt', map_location='cpu')
    train_tensors = data['x']
    train_labels = data['y']
    
    for i in range(10):
        t = train_tensors[i]
        lbl = train_labels[i].item()
        print(f"Tile {i} (Label: {lbl}): Min: {t.min().item():.3f}, Max: {t.max().item():.3f}, Mean: {t.mean().item():.3f}, Std: {t.std().item():.3f}")

if __name__ == "__main__":
    main()
