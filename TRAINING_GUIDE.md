# 🎯 Chessimg2pos: Training Guide (Industrial Edition)

This guide covers the **Tensor Beast** pipeline—the most efficient way to train high-accuracy chess recognition models.

---

## 🚀 The Core Workflow (Zero-Latency)

We no longer train directly from individual PNG files. Training from millions of tiny files causes a massive CPU bottleneck. Instead, we use a **two-step pipeline**:

### Step 1: Data Preparation (Choose One Path)

#### A. Synthetic Path (Best for Chess-Puzzle Bots) ⭐ RECOMMENDED
Generates perfect Lichess-style boards with varied themes directly into optimized tensors.
```bash
# Generates ~3.2M tiles in ~10 minutes
python3 generate_lichess_tensors.py
```

#### B. Real-World Path (For Physical Photos)
Packs your existing PNG tiles into optimized tensors, using **Board-Level Splitting** to prevent data leakage.
```bash
# Packs existing images/tiles_real/ into tensor_dataset_safe/
python3 tensor_packer_kaggle_dataset.py
```

---

### Step 2: High-Speed Training

Once your tensors are ready in `tensor_dataset_synthetic/` or `tensor_dataset_safe/`, use the Beast script for maximum speed.

```bash
# 🐉 KAGGLE TENSOR BEAST (Target: 99.9% accuracy)
python3 train_kaggle_tensor_beast.py --data-dir tensor_dataset_synthetic --epochs 50
```

---

## ⚡ Key Scripts Reference

| Script | Model Class | Best For |
| :--- | :--- | :--- |
| `train_kaggle_tensor_beast.py` | `UltraEnhanced` | **Production Models**. Best accuracy, high diversity. |
| `train_kaggle_tensor_enhanced.py`| `Enhanced` | Balanced performance and speed. |
| `train_kaggle_tensor_basic.py` | `Standard` | Extremely fast training, lightweight models. |
| `generate_lichess_tensors.py` | N/A | Generating infinite, perfect Lichess datasets. |
| `tensor_packer_kaggle_dataset.py`| N/A | Converting PNGs to high-speed binary tensors. |

---

## 📊 Why Use Tensors?

*   **Speed**: Training goes from **1 hour/epoch** to **3 minutes/epoch**.
*   **Zero Bottlenecks**: Data is loaded into System RAM (30GB on Kaggle) once at startup.
*   **VRAM Safe**: Data is moved to the GPU only at the batch level, preventing OOM.
*   **Augmentation**: Random rotation, brightness, and noise are applied **on the GPU** during training.

---

## 🛠️ Performance Tuning (CLI Arguments)

All tensor scripts support the following arguments:
- `--data-dir`: Folder containing `.pt` chunks (default: `tensor_dataset_synthetic`).
- `--epochs`: Number of epochs (default: 50).
- `--batch-size`: Images per step (default: 4096).
- `--lr`: Learning rate (default: 0.001).

---

## 📉 Expected Results

- **Synthetic Data**: Accuracy usually hits **99.9%** within 10-15 epochs.
- **Real-World Data**: Accuracy starts lower (due to the safe split) but produces much better real-world results than legacy scripts.

---

## ⚠️ Legacy Scripts (DO NOT USE FOR LARGE DATASETS)
The following scripts are slow and prone to OOM or data leakage:
- `scripts/train_kaggle_beast.py`
- `scripts/train_kaggle_optimized.py`
- `scripts/train_kaggle_stream_turbo.py` (The streaming experiment)
