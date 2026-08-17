#!/usr/bin/env bash
# Fetch the KITTI odometry sequence 00 subset (500 frames, ~622MB) referenced by BENCHMARK.md
# (google drive share from the small_gicp README; CC BY-NC-SA 3.0 — non-commercial use only).
set -e

DEST=${1:-$HOME/datasets/kitti/odometry}
FILE_ID="1h9tARKvX6BwLfc_vfMfdxuP3bSbmgjmd"
mkdir -p "$DEST"

if command -v gdown >/dev/null 2>&1; then
  :
elif command -v pip3 >/dev/null 2>&1; then
  pip3 install -q gdown
fi

echo "Downloading KITTI00.tar.gz (~622MB) to $DEST ..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$DEST/KITTI00.tar.gz"

echo "Extracting ..."
tar xzf "$DEST/KITTI00.tar.gz" -C "$DEST"

echo "Done. Frames:"
find "$DEST" -name "*.bin" | wc -l
