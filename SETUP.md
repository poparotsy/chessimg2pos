# Chess Image to Position - Setup Guide

## Fresh Installation (New Server/Machine)

### Prerequisites
- Python 3.10 or higher
- pip

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/mdicio/chessimg2pos.git
cd chessimg2pos

# 2. Install the package
pip install chessimg2pos

# 3. Copy the chess_recognizer.py script to your project
# (The script is in this repo)

# 4. Test it
python3 chess_recognizer.py ./images/chess_image_2.png
```

## Alternative: Install from this directory

```bash
# If you're bundling this exact code
cd /path/to/chessimg2pos
pip install -e .
```

## What Gets Installed

The package will install these dependencies:
- torch (PyTorch - deep learning framework)
- torchvision
- numpy
- matplotlib
- Pillow (image processing)
- pandas
- requests (for downloading the pretrained model)

## First Run

On first run, the script will automatically:
1. Download the pretrained model (~600KB) from GitHub releases
2. Cache it locally for future use

## For Your Twitter Bot

Just ensure:
1. `pip install chessimg2pos` is in your deployment script
2. Include `chess_recognizer.py` in your repo
3. Call it from JavaScript as shown in `example_usage.js`

## Minimal Docker Example

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install chessimg2pos
COPY chess_recognizer.py .
# Your bot code here
```
