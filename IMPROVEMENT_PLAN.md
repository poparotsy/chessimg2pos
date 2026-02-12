# Chess Image Recognition - Improvement Plan

## Current Status

✅ **Working:** Enhanced model (`chess_recognizer.py`) - decent accuracy  
❌ **Problem:** Confuses Queens with Rooks on some images  
⚠️ **Attempted:** Ultra model - more complex but worse accuracy (overfitting)

## The Issue

The model struggles to distinguish Queens from Rooks because:
1. Training data may not have enough queen/rook examples
2. Queens and rooks look similar in some chess board styles
3. Need more diverse training data

## Solution: Augment Training Data

### Step 1: Generate More Training Images (Focused on Queens/Rooks)

```bash
# Install dependencies
pip install python-chess cairosvg pillow

# Generate 500 new chess boards with emphasis on queens and rooks
python3 generate_real_boards.py
```

This creates images in: `images/chessboards/generated_qr/`

### Step 2: Combine with Existing Data

```bash
# The generate_all_tiles.py script will process ALL images including new ones
python3 generate_all_tiles.py
```

This creates tiles from all board images (including the new 500) in: `images/tiles_all/`

### Step 3: Retrain the Model

```bash
# Train ultra model with the augmented dataset
python3 train_with_existing_tiles.py
```

This will:
- Use all tiles (old + new queen/rook focused ones)
- Train for 15 epochs
- Save to `models/model_ultra_all.pt`
- Take 20-40 minutes

### Step 4: Update the Recognizer

Edit `chess_recognizer_ultra.py` line 25 to use the new model:
```python
predictor = ChessPositionPredictor("./models/model_ultra_all.pt", classifier="ultra")
```

### Step 5: Test

```bash
python3 chess_recognizer_ultra.py your_problem_image.png
```

Compare with enhanced model:
```bash
python3 chess_recognizer.py your_problem_image.png
```

## Alternative: Find More Training Data Online

Instead of generating, download real chess position images:

1. **Lichess Puzzle Database**
   - https://database.lichess.org/#puzzles
   - Download puzzle images
   - Place in `images/chessboards/lichess_puzzles/`

2. **Chess.com Positions**
   - Screenshot various positions
   - Focus on endgames with queens and rooks

3. **SCID Database**
   - Export positions as images
   - Add to training set

## Current Files

### Working Scripts
- `chess_recognizer.py` - Enhanced model (currently best)
- `chess_recognizer_ultra.py` - Ultra model (needs better training data)
- `generate_real_boards.py` - Generate synthetic training boards
- `generate_all_tiles.py` - Extract tiles from all board images
- `train_with_existing_tiles.py` - Train ultra model

### Models
- `models/model_enhanced.pt` - Current best model
- `models/model_ultra.pt` - Trained on limited data (poor)
- `models/model_ultra_all.pt` - Will be created after retraining

## For Your Twitter Bot

**Current recommendation:** Use `chess_recognizer.py` (enhanced model)

**After retraining:** Test both and use whichever is more accurate

## Quick Commands Summary

```bash
# 1. Generate more training data
pip install python-chess cairosvg pillow
python3 generate_real_boards.py

# 2. Process all images into tiles
python3 generate_all_tiles.py

# 3. Retrain with augmented data
python3 train_with_existing_tiles.py

# 4. Test the new model
python3 chess_recognizer_ultra.py test_image.png

# 5. Compare with current model
python3 chess_recognizer.py test_image.png
```

## Expected Outcome

With 500+ additional queen/rook focused training images:
- Model should better distinguish queens from rooks
- Average confidence should remain high (>85%)
- Accuracy on your problem images should improve

## Notes

- Training takes 20-40 minutes
- More training data = better accuracy (up to a point)
- Can repeat Step 1 to generate even more boards (1000+)
- Monitor training accuracy - should reach 95%+ on validation set

---

**Tomorrow:** Run the commands above and test on your problematic images!
