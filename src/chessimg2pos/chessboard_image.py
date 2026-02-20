import numpy as np
import PIL.Image


def _get_resized_chessboard(chessboard_img_path):
    """chessboard_img_path = path to a chessboard image
    Returns a 256x256 image of a chessboard (32x32 per tile)
    """
    img_data = PIL.Image.open(chessboard_img_path).convert("RGB")
    return img_data.resize([256, 256], PIL.Image.BILINEAR)


def get_chessboard_tiles(chessboard_img_path, use_grayscale=True):
    """chessboard_img_path = path to a chessboard image or PIL object
    use_grayscale = true/false for whether to return tiles in grayscale

    Returns a list (length 64) of 32x32 image data
    """
    if isinstance(chessboard_img_path, str):
        img_data = PIL.Image.open(chessboard_img_path).convert("RGB")
    else:
        img_data = chessboard_img_path.convert("RGB")
        
    if use_grayscale:
        img_data = img_data.convert("L")  # Standard L conversion
        
    # We assume the chessboard covers the whole image
    # We slice it using dynamic tile sizes first, and then resize the TILES to 32x32
    w, h = img_data.size
    tile_w = w / 8.0
    tile_h = h / 8.0
    
    tiles = [None] * 64
    for rank in range(8):  # rows/ranks (numbers)
        for file in range(8):  # columns/files (letters)
            sq_i = rank * 8 + file
            
            # Crop exact coordinates
            left = file * tile_w
            upper = rank * tile_h
            right = (file + 1) * tile_w
            lower = (rank + 1) * tile_h
            
            tile = img_data.crop((left, upper, right, lower))
            # Resize tile to 32x32 properly without stretching the whole board upfront
            tile = tile.resize((32, 32), PIL.Image.BILINEAR)
            
            # Convert back to RGB format if that's what's expected for down stream (although prediction wants L if Grayscale)
            # We'll just return the tile and let predictor.py handle it, 
            # predictor.py explicitly converts it based on use_grayscale 
            tiles[sq_i] = tile
            
    return tiles
