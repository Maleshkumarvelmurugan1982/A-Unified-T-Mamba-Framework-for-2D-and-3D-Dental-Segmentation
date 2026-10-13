#!/bin/bash
# Downloads + verifies the dataset (NC-release-data-checked/{train,valid}/{images,labels}/*.nii.gz) and your baseline checkpoint. Safe to re-run.
# Handles .tar / .tar.gz / .zip archives and one level of extra folder nesting.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DATA_ID=${DATA_ID:-1ivIhA222AoFxzTYWVAC3yUFSsg5Waqqu}
CKPT_ID=${CKPT_ID:-1WQeyZLiUeiNm8onicyeB5-TipOPZuaHu}
source $CONDA/etc/profile.d/conda.sh; conda activate tmamba
mkdir -p "$BASE/datasets" "$REPO/pretrain"
if [ ! -d "$DATA/train/images" ]; then
  echo "===== DOWNLOADING DATASET ====="
  TMPF="$BASE/dataset_download.bin"
  gdown "$DATA_ID" -O "$TMPF"
  file "$TMPF"
  echo "===== EXTRACTING ====="
  EX="$BASE/datasets/_extract"; rm -rf "$EX"; mkdir -p "$EX"
  if file "$TMPF" | grep -qi zip; then unzip -q "$TMPF" -d "$EX"; else tar -xf "$TMPF" -C "$EX"; fi
  rm -f "$TMPF"
  SRC=$(find "$EX" -maxdepth 3 -type d -name train | head -1)
  [ -n "$SRC" ] || { echo "ERROR: no 'train' folder found inside the archive"; exit 1; }
  SRC=$(dirname "$SRC")
  rm -rf "$DATA"; mv "$SRC" "$DATA"; rm -rf "$EX"
else
  echo "Dataset already present."
fi
rm -rf "$DATA/sub_volumes"   # stale cache from earlier runs would point at old paths
for sp in train valid; do
  ni=$(ls "$DATA/$sp/images" | wc -l); nl=$(ls "$DATA/$sp/labels" | wc -l)
  echo "$sp: $ni images / $nl labels"
  [ "$ni" -gt 0 ] && [ "$ni" = "$nl" ] || { echo "ERROR: dataset check failed for $sp"; exit 1; }
done
echo "DATASET OK -> $DATA"
if [ ! -f "$REPO/pretrain/best_TMamba3D.pth" ]; then
  gdown "$CKPT_ID" -O "$REPO/pretrain/best_TMamba3D.pth" || echo "(baseline checkpoint download failed - download it manually into pretrain/best_TMamba3D.pth)"
fi
ls -la "$REPO/pretrain/"
