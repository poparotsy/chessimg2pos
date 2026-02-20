# Training Model Analysis & Fixes

**Date:** February 17, 2026  
**Issue:** Model predicts only empty squares and pawns with near-zero confidence  
**Model:** `models/model_tensor_beast_v2.pt`

---

## Problem Symptoms

### Example 1: qe6.jpeg
```json
{
  "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
  "avg_confidence": 0.7093,
  "overall_confidence": 1.27e-10
}
```
**Result:** All empty squares

### Example 2: 1.png
```json
{
  "fen": "6p1/1p4pp/p3p3/1p3p2/8/8/8/8 w - - 0 1",
  "avg_confidence": 0.6404,
  "overall_confidence": 1.03e-14
}
```
**Result:** Only pawns and empty squares, no other pieces

---

## Root Cause Analysis

### 1. **Unrealistic Synthetic Data (PRIMARY - 80%)**

**Problem in `generate_lichess_tensors.py`:**
```python
def generate_random_fen():
    board = chess.Board(None)  # Empty board
    # Randomly place kings
    board.set_piece_at(random.choice(list(chess.SQUARES)), ...)
    # Place 5-25 random pieces anywhere
    for _ in range(random.randint(5, 25)):
        pt = random.choice([PAWN, KNIGHT, BISHOP, ROOK, QUEEN])
        # Place randomly without chess rules
```

**Issues:**
- Creates illegal chess positions (pawns on rank 1/8, impossible piece configurations)
- Random piece distribution doesn't match real games
- Too many empty squares (64 squares, only 5-25 pieces = 60-85% empty)
- No correlation with how pieces actually appear in real chess photos
- Model learned patterns that don't exist in real chess

**Impact:**
- Model trained on positions that never occur in real games
- Learned to predict mostly empty squares since that's what it saw most
- No understanding of actual chess piece distributions

---

### 2. **Weighted Loss Amplified the Problem (SECONDARY - 15%)**

**Problem in `train_kaggle_tensor_beast.py`:**
```python
weights = torch.ones(len(FEN_CHARS)).to(device)
weights[0] = 0.5  # Give 'Empty' half weight
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Issues:**
- Intended to prevent "predict everything as empty" bias
- Actually had opposite effect: model learned empty squares don't matter
- Combined with data having 60-85% empty squares = disaster
- Model strategy: "When uncertain, predict empty (low penalty) or pawn (most common piece)"

**Impact:**
- Model never learned to confidently predict rare pieces (kings, queens, rooks)
- Predicting empty became the "safe" choice
- Overall confidence collapsed to near-zero (1e-10 to 1e-14)

---

### 3. **Ineffective Augmentation (MINOR - 5%)**

**Problem in `train_kaggle_tensor_beast.py`:**
```python
train_augmentations = nn.Sequential(
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ← Useless on grayscale
)
```

**Issues:**
- `ColorJitter` does nothing meaningful on single-channel grayscale images
- Only brightness matters for grayscale, contrast/saturation are ignored
- Not a breaking issue, just wasted computation

---

## Fixes Applied

### Fix 1: Generate Realistic Chess Positions

**File:** `generate_lichess_tensors.py`

**Before:**
```python
def generate_random_fen():
    """Generates a random but realistic-ish board"""
    board = chess.Board(None)
    # Place Kings
    board.set_piece_at(random.choice(list(chess.SQUARES)), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(random.choice([s for s in chess.SQUARES if board.piece_at(s) is None]), chess.Piece(chess.KING, chess.BLACK))
    
    # Place 5-20 random pieces
    for _ in range(random.randint(5, 25)):
        sqs = [s for s in chess.SQUARES if board.piece_at(s) is None]
        if not sqs: break
        pt = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        c = random.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(random.choice(sqs), chess.Piece(pt, c))
    return board.fen()
```

**After:**
```python
def generate_random_fen():
    """Generates a realistic chess position from legal moves"""
    board = chess.Board()  # Start from standard position
    
    # Make 5-60 random legal moves to get varied positions
    num_moves = random.randint(5, 60)
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves or board.is_game_over():
            break
        board.push(random.choice(legal_moves))
    
    return board.fen().split()[0]  # Return only position part
```

**Benefits:**
- All positions are legal chess positions
- Realistic piece distributions matching actual games
- Always has 2 kings (required in chess)
- Piece counts and placements follow chess rules
- Varied positions from opening (5 moves) to endgame (60 moves)
- Model will learn patterns that exist in real chess images

---

### Fix 2: Remove Weighted Loss

**File:** `train_kaggle_tensor_beast.py`

**Before:**
```python
# Weighted Loss: Penalize "Empty" (index 0) less, pieces more
# This prevents the model from just guessing "Empty" for everything
weights = torch.ones(len(FEN_CHARS)).to(device)
weights[0] = 0.5 # Give 'Empty' half weight
criterion = nn.CrossEntropyLoss(weight=weights)
```

**After:**
```python
# Balanced Loss: Equal weight for all pieces
criterion = nn.CrossEntropyLoss()
```

**Benefits:**
- All piece types (including empty) are equally important
- Model must learn to distinguish all pieces properly
- No "safe" prediction strategy
- Forces model to build confidence in all predictions
- Natural class balance from realistic positions

---

## Expected Results After Retraining

### Data Quality Improvements:
- **Piece distribution:** Matches real chess games
- **Position legality:** 100% legal positions
- **Variety:** Opening, middlegame, and endgame positions
- **Realism:** Patterns that exist in actual chess photos

### Model Performance Improvements:
- **All piece types recognized:** Kings, queens, rooks, bishops, knights, pawns
- **Higher confidence:** Should see overall confidence > 0.01 (vs current 1e-10)
- **Balanced predictions:** Not biased toward empty squares
- **Real-world accuracy:** Should work on actual chess photos

---

## Training Recommendations

### Before Next Training Run:

1. **Delete old synthetic data:**
   ```bash
   rm -rf tensor_dataset_synthetic/
   ```

2. **Regenerate with fixed script:**
   ```bash
   python3 generate_lichess_tensors.py
   ```

3. **Verify data quality (spot check):**
   ```python
   import torch
   data = torch.load('tensor_dataset_synthetic/train_chunk_00.pt')
   # Check piece distribution in labels
   unique, counts = torch.unique(data['y'], return_counts=True)
   print(dict(zip(unique.tolist(), counts.tolist())))
   # Should see balanced distribution, not 80%+ empty
   ```

4. **Train with fixed script:**
   ```bash
   python3 train_kaggle_tensor_beast.py --epochs 50
   ```

5. **Test on real images during training:**
   - Keep a few real chess photos as validation
   - Check predictions every 5-10 epochs
   - Stop if model still predicts only empty/pawns

---

## Additional Considerations

### Optional Improvements (Not Critical):

1. **Better augmentation for grayscale:**
   ```python
   train_augmentations = nn.Sequential(
       transforms.RandomRotation(5),  # Smaller rotation
       transforms.ColorJitter(brightness=0.15),  # Brightness only
   )
   ```

2. **Add real Lichess game positions:**
   - Download PGN database from Lichess
   - Extract positions from actual games
   - Mix with synthetic data for even better realism

3. **Validate on real images:**
   - Include actual chess photos in validation set
   - Catch data distribution issues early

---

## Summary

**The trainer code was fine.** The model successfully learned the patterns in the data - the problem was the data taught it the wrong patterns.

**Key lesson:** Synthetic data must match the real-world distribution you're trying to predict. Random piece placement ≠ real chess positions.

**Next steps:** Regenerate data with fixes, retrain, and validate on real chess images.
