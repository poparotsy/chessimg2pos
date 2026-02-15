#!/bin/bash
# Safe cleanup - moves files to archive instead of deleting

cd "$(dirname "$0")"

echo "🧹 Starting cleanup..."
echo ""

# Create archive directory
mkdir -p archive/training_scripts
mkdir -p archive/notebooks
mkdir -p archive/generators
mkdir -p archive/misc

echo "📦 Moving old training scripts to archive..."
mv scripts/train_80k_beast_v4.py archive/training_scripts/ 2>/dev/null
mv scripts/train_real_80k*.py archive/training_scripts/ 2>/dev/null
mv scripts/train_with_existing_tiles.py archive/training_scripts/ 2>/dev/null
mv scripts/train_ultra*.py archive/training_scripts/ 2>/dev/null
mv scripts/train_from_dataset.py archive/training_scripts/ 2>/dev/null
mv scripts/train_3d_rendered.py archive/training_scripts/ 2>/dev/null

echo "📓 Moving notebooks to archive..."
mv scripts/*.ipynb archive/notebooks/ 2>/dev/null

echo "🔧 Moving generators to archive..."
mv scripts/generate_*.py archive/generators/ 2>/dev/null

echo "🗂️  Moving misc scripts to archive..."
mv scripts/chess_recognizer*.py archive/misc/ 2>/dev/null
mv colab_resume.tips archive/misc/ 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 Remaining in scripts/:"
ls -1 scripts/
echo ""
echo "📦 Archived files in archive/"
echo ""
echo "To restore: mv archive/training_scripts/* scripts/"
echo "To delete archive: rm -rf archive/"
