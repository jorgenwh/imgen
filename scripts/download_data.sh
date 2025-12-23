#!/bin/bash
# Download and prepare datasets for training experiments.

set -e

DATA_DIR="./data"

echo "=== MNIST ==="
mkdir -p "$DATA_DIR"
echo "  (Auto-downloads on first training run)"

echo ""
echo "=== COCO ==="
COCO_DIR="$DATA_DIR/coco"
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

# Annotations
if [ ! -f "annotations/captions_train2017.json" ]; then
    echo "  Downloading annotations..."
    curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "  Extracting annotations..."
    unzip -q annotations_trainval2017.zip
else
    echo "  Annotations already exist"
fi

# Val images
if [ ! -d "val2017" ]; then
    echo "  Downloading val2017..."
    curl -O http://images.cocodataset.org/zips/val2017.zip
    echo "  Extracting val2017..."
    unzip -q val2017.zip
else
    echo "  val2017 already exists"
fi

# Train images
if [ ! -d "train2017" ]; then
    echo "  Downloading train2017 (~18GB, this will take a while)..."
    curl -O http://images.cocodataset.org/zips/train2017.zip
    echo "  Extracting train2017..."
    unzip -q train2017.zip
else
    echo "  train2017 already exists"
fi

echo ""
echo "=== Cleanup ==="
echo "  Removing zip files..."
rm -f *.zip

echo ""
echo "=== Summary ==="
echo "  train2017: $(ls train2017 | wc -l) images"
echo "  val2017: $(ls val2017 | wc -l) images"
echo ""
echo "Done!"
