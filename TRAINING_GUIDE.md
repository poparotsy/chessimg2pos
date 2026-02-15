# 🎯 Training Scripts - Quick Reference

## ⚡ Use These Scripts

### 🐉 **Kaggle BEAST Mode** (Dual T4 GPUs) ⭐ RECOMMENDED
```bash
python3 train_kaggle_beast.py
```
- Batch size: 4096 (maximum GPU utilization)
- Mixed precision (AMP) + Gradient Scaling
- Focal Loss + Label Smoothing
- Real-time progress bars with ETA
- Dual GPU support (DataParallel)
- 25 epochs
- Uses `UltraEnhancedChessPieceClassifier`

### 🏆 **Kaggle Standard** (Dual T4 GPUs)
```bash
python3 train_kaggle_optimized.py
```
- Batch size: 512
- Mixed precision (AMP)
- Dual GPU support
- 25 epochs
- Uses `UltraEnhancedChessPieceClassifier`

### 💻 **Local Training** (ARM/x86/CUDA/MPS)
```bash
python3 train_local_optimized.py
```
- Batch size: 128
- Apple Silicon (MPS) compatible
- CUDA/CPU fallback
- 20 epochs
- Uses `UltraEnhancedChessPieceClassifier`

---

## 📊 Both Scripts Include

✅ **Board-level validation split** (reliable accuracy)  
✅ **Checkpoint resumption** (auto-resume on crash)  
✅ **Progress logging** (ETA, loss, accuracy)  
✅ **Best model saving** (keeps highest validation accuracy)  
✅ **Gradient clipping** (training stability)  
✅ **Learning rate scheduling**

---

## 🗂️ Output Files

### Kaggle
- Model: `models/model_kaggle_ultra.pt`
- Checkpoint: `models/checkpoint_kaggle_ultra.pt`

### Local
- Model: `models/model_local_ultra.pt`
- Checkpoint: `models/checkpoint_local_ultra.pt`

---

## 🧹 Scripts Directory Cleanup

See `scripts/README.md` for full documentation.

**Keep**: 
- `train_kaggle_optimized.py` (root)
- `train_local_optimized.py` (root)
- `scripts/download_datasets.py`
- `scripts/prepare_all_datasets.py`
- `scripts/prepare_real_data.py`
- `scripts/ensemble_predictor.py`
- `scripts/test_model.py`

**Archive/Delete**: All other training variants (superseded)

---

## 🚀 Quick Start

```bash
# 1. Prepare data (if needed)
python3 scripts/prepare_all_datasets.py

# 2. Train (choose one)
python3 train_kaggle_optimized.py    # Kaggle
python3 train_local_optimized.py     # Local

# 3. Test
python3 scripts/test_model.py
```

---

## 🔍 Model Architecture

**UltraEnhancedChessPieceClassifier** (Best Available)
- Attention mechanism
- Dual pooling (avg + max)
- Batch normalization
- Dropout regularization
- 13 output classes

---

## 💡 Performance Tips

### Kaggle
- Enable GPU accelerator (T4 x2)
- Increase batch size if more VRAM available
- Use `num_workers=4` (4 vCPUs)

### Local
- Script auto-detects best device
- Reduce batch size if OOM errors
- Use `num_workers=2` or `0` for CPU

---

## 📈 Expected Results

- **Training accuracy**: ~99%+
- **Validation accuracy**: ~97-98%
- **Training time**: 
  - Kaggle (T4 x2): ~2-3 hours
  - Local (varies): 4-12 hours

---

## 🐛 Troubleshooting

**OOM Error**: Reduce `BATCH_SIZE` in script  
**Slow training**: Check GPU usage with `nvidia-smi`  
**No GPU detected**: Verify CUDA/MPS installation  
**Checkpoint issues**: Delete checkpoint file to restart fresh
