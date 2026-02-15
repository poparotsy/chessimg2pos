# 📁 Scripts Directory - Documentation

## 🚀 RECOMMENDED SCRIPTS (Use These!)

### Training Scripts

#### **`../train_kaggle_optimized.py`** ⭐ BEST FOR KAGGLE
- **Purpose**: Optimized for Kaggle dual T4 GPUs (30GB VRAM)
- **Features**:
  - Dual GPU support (DataParallel)
  - Mixed precision training (AMP)
  - Board-level validation split
  - Checkpoint resumption
  - Uses `UltraEnhancedChessPieceClassifier` (best model)
- **Config**: Batch size 512, 25 epochs, 4 workers
- **Usage**: `python3 train_kaggle_optimized.py`

#### **`../train_local_optimized.py`** ⭐ BEST FOR LOCAL
- **Purpose**: Optimized for local hardware (ARM/x86/CUDA/MPS)
- **Features**:
  - Apple Silicon (MPS) support
  - CUDA support
  - CPU fallback
  - Board-level validation split
  - Checkpoint resumption
  - Uses `UltraEnhancedChessPieceClassifier` (best model)
- **Config**: Batch size 128, 20 epochs, 2 workers
- **Usage**: `python3 train_local_optimized.py`

---

## 📊 Data Preparation Scripts

### **`download_datasets.py`**
- Downloads chess datasets from Kaggle
- Requires Kaggle API credentials
- **Usage**: `python3 scripts/download_datasets.py`

### **`prepare_all_datasets.py`**
- Prepares and organizes all downloaded datasets
- Generates tiles from board images
- **Usage**: `python3 scripts/prepare_all_datasets.py`

### **`prepare_real_data.py`**
- Prepares real chess board images
- Extracts tiles organized by board
- **Usage**: `python3 scripts/prepare_real_data.py`

---

## 🧪 Testing & Prediction Scripts

### **`test_model.py`**
- Quick model testing on sample images
- **Usage**: `python3 scripts/test_model.py`

### **`ensemble_predictor.py`**
- Ensemble prediction using multiple models
- Improves accuracy by combining predictions
- **Usage**: See script for API usage

---

## 🗂️ Legacy/Experimental Scripts (Archive)

These scripts are kept for reference but are superseded by the optimized versions:

### Training Variants
- `train_real_80k.py` - Original 80k training
- `train_real_80k_resumable.py` - Added resumption
- `train_real_80k_resumable_v2.py` - Improved resumption
- `train_real_80k_resumable_v3_kaggle.py` - Kaggle variant (superseded)
- `train_real_80k_board_split.py` - Board-level split experiment
- `train_finetune_real.py` - Fine-tuning experiments
- `train_2d_combined.py` - 2D dataset training
- `train_lichess_2d.py` - Lichess-specific training
- `train_ultra_fresh.py` - Ultra model experiments
- `train_ultra_model.py` - Ultra model variant
- `train_ultra_from_tiles.py` - Ultra from pre-generated tiles
- `train_ultra_full.py` - Full ultra training
- `train_3d_rendered.py` - 3D rendered boards
- `train_from_dataset.py` - Generic dataset training
- `train_with_existing_tiles.py` - Pre-existing tiles

### Data Generation (Legacy)
- `generate_tiles_from_dataset.py` - Generic tile generation
- `generate_tiles_3d.py` - 3D tile generation
- `generate_tiles_source.py` - Source-specific tiles
- `generate_tiles_lichess.py` - Lichess tiles
- `generate_all_tiles.py` - Batch tile generation
- `generate_lichess_data.py` - Lichess data generation
- `generate_real_boards.py` - Real board generation
- `generate_queen_rook_data.py` - Specific piece data

### Notebooks (Legacy)
- `colab_train.ipynb` - Google Colab training
- `colab_train_3d.ipynb` - Colab 3D training
- `colab_train_drive.ipynb` - Colab with Drive

### Other
- `chess_recognizer.py` - Basic recognizer
- `chess_recognizer_ultra.py` - Ultra recognizer
- `test_2d_model.py` - 2D model testing
- `test_chessimg2pos.py` - Package testing

---

## 🧹 Cleanup Recommendations

### Keep These:
- `../train_kaggle_optimized.py` ⭐
- `../train_local_optimized.py` ⭐
- `download_datasets.py`
- `prepare_all_datasets.py`
- `prepare_real_data.py`
- `ensemble_predictor.py`
- `test_model.py`

### Archive/Delete These:
All other training scripts are superseded by the two optimized versions.

---

## 📝 Quick Start Guide

### 1. Prepare Data
```bash
# Download datasets (if needed)
python3 scripts/download_datasets.py

# Prepare all data
python3 scripts/prepare_all_datasets.py
```

### 2. Train Model

**On Kaggle:**
```bash
python3 train_kaggle_optimized.py
```

**Locally:**
```bash
python3 train_local_optimized.py
```

### 3. Test Model
```bash
python3 scripts/test_model.py
```

---

## 🔧 Model Architecture

All optimized scripts use **`UltraEnhancedChessPieceClassifier`**:
- Attention mechanism
- Dual pooling (avg + max)
- Batch normalization
- Dropout regularization
- 13 classes (FEN characters: `1RNBQKPrnbqkp`)

---

## 💡 Tips

- **Kaggle**: Enable GPU in notebook settings (T4 x2 recommended)
- **Local**: Script auto-detects CUDA/MPS/CPU
- **Resumption**: Both scripts auto-resume from checkpoints
- **Monitoring**: Watch GPU usage with `nvidia-smi` (CUDA) or Activity Monitor (Mac)
- **Batch Size**: Increase if you have more VRAM, decrease if OOM errors occur

---

## 📞 Support

For issues or questions, check:
1. Main README.md
2. IMPROVEMENT_PLAN.md
3. GitHub issues
