# 🧹 Cleanup Plan for GitHub Repo

## ✅ KEEP (Essential Files)

### Root Level
- `README.md` - Main documentation
- `TRAINING_GUIDE.md` - Quick reference
- `SETUP.md` - Setup instructions
- `IMPROVEMENT_PLAN.md` - Development roadmap
- `DOCUMENTATION.md` - Detailed docs
- `requirements.txt` - Dependencies
- `pyproject.toml` - Package config
- `MANIFEST.in` - Package manifest
- `LICENCE` - License file
- `.gitignore` - Git ignore rules

### Training Scripts (Root)
- `train_kaggle_beast.py` ⭐ PRIMARY - Kaggle optimized
- `train_kaggle_optimized.py` - Kaggle standard
- `train_local_optimized.py` - Local training

### Directories
- `src/` - Source code (keep all)
- `examples/` - Usage examples (keep all)
- `models/` - Trained models (gitignored)
- `images/` - Sample images (keep)

### Scripts Directory (Keep These)
- `scripts/README.md` - Documentation
- `scripts/prepare_all_datasets.py` - Data prep
- `scripts/test_model.py` - Testing

---

## ❌ DELETE (Superseded/Redundant)

### Training Scripts (scripts/)
```bash
# All superseded by train_kaggle_beast.py and train_local_optimized.py
scripts/train_80k_beast_v4.py
scripts/train_real_80k_resumable_v3_kaggle.py
scripts/train_real_80k_resumable_v2.py
scripts/train_real_80k_resumable.py
scripts/train_real_80k_board_split.py
scripts/train_real_80k.py
scripts/train_with_existing_tiles.py
scripts/train_ultra_model.py
scripts/train_ultra_from_tiles.py
scripts/train_ultra_full.py
scripts/train_from_dataset.py
scripts/train_3d_rendered.py
```

### Notebooks (scripts/)
```bash
# Colab notebooks - move to separate examples/ if needed
scripts/colab_train_drive.ipynb
scripts/colab_train.ipynb
scripts/colab_train_3d.ipynb
```

### Data Generation (scripts/)
```bash
# Keep only if actively used, otherwise delete
scripts/generate_queen_rook_data.py
scripts/generate_tiles_from_dataset.py
scripts/generate_all_tiles.py
scripts/generate_tiles_3d.py
scripts/generate_real_boards.py
```

### Recognizer Scripts (scripts/)
```bash
# Redundant - functionality in src/
scripts/chess_recognizer_ultra.py
scripts/chess_recognizer.py
```

### Other
```bash
colab_resume.tips  # Move to docs or delete
__pycache__/  # Should be gitignored
```

---

## 🔧 Cleanup Commands

### Safe Cleanup (Move to archive/)
```bash
cd chessimg2pos
mkdir -p archive/training_scripts
mkdir -p archive/notebooks
mkdir -p archive/generators

# Move old training scripts
mv scripts/train_*.py archive/training_scripts/

# Move notebooks
mv scripts/*.ipynb archive/notebooks/

# Move generators (optional)
mv scripts/generate_*.py archive/generators/

# Move recognizers
mv scripts/chess_recognizer*.py archive/
```

### Aggressive Cleanup (Delete)
```bash
cd chessimg2pos/scripts

# Delete old training scripts
rm train_80k_beast_v4.py
rm train_real_80k*.py
rm train_with_existing_tiles.py
rm train_ultra*.py
rm train_from_dataset.py
rm train_3d_rendered.py

# Delete notebooks
rm *.ipynb

# Delete redundant scripts
rm chess_recognizer*.py

# Optional: Delete generators if not used
rm generate_*.py
```

---

## 📁 Final Structure

```
chessimg2pos/
├── README.md
├── TRAINING_GUIDE.md
├── SETUP.md
├── requirements.txt
├── pyproject.toml
├── train_kaggle_beast.py ⭐
├── train_kaggle_optimized.py
├── train_local_optimized.py
├── src/
│   └── chessimg2pos/
│       ├── __init__.py
│       ├── chessclassifier.py
│       ├── chessdataset.py
│       ├── predictor.py
│       ├── trainer.py
│       └── ...
├── scripts/
│   ├── README.md
│   ├── prepare_all_datasets.py
│   └── test_model.py
├── examples/
│   └── demo_usage.ipynb
└── images/
    └── (sample images)
```

---

## 🎯 Recommended Action

**Option 1: Safe (Archive)**
```bash
cd /Users/guru/workspace/playground/chessai/py/chessimg2pos/chessimg2pos
mkdir -p archive
mv scripts/train_*.py archive/ 2>/dev/null || true
mv scripts/*.ipynb archive/ 2>/dev/null || true
mv scripts/chess_recognizer*.py archive/ 2>/dev/null || true
mv scripts/generate_*.py archive/ 2>/dev/null || true
```

**Option 2: Clean (Delete)**
```bash
cd /Users/guru/workspace/playground/chessai/py/chessimg2pos/chessimg2pos/scripts
rm train_*.py
rm *.ipynb
rm chess_recognizer*.py
# Keep generate_* if you use them, otherwise:
# rm generate_*.py
```

---

## ✅ After Cleanup

Update `.gitignore`:
```
__pycache__/
*.pyc
models/*.pt
images/tiles_*
archive/
.DS_Store
```

Commit:
```bash
git add .
git commit -m "Clean up redundant training scripts - use train_kaggle_beast.py"
git push
```
