#!/bin/bash
# Downloads the 2D dental X-ray dataset (Tooth2D-X-Ray-6k: train/{images,annotations}, test/{images,annotations}). Safe to re-run.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source $CONDA/etc/profile.d/conda.sh; conda activate tmamba
mkdir -p "$BASE/datasets" "$BASE/dataset_downloads"
if [ ! -d "$DATA2D/train/images" ]; then
  cd "$BASE/dataset_downloads"
  [ -s dataset_part_1 ] || gdown --id 1YTa8Z5iv_pO0G-0KeU-yTritkodg7TMK -O dataset_part_1
  [ -s dataset_part_2 ] || gdown --id 1CUq-MuMIr7eHKBVZ4VTh-m72pcLMhgp2 -O dataset_part_2
  mkdir -p extracted
  extract() { case "$(file -b "$1")" in
      *Zip\ archive*) unzip -o -q "$1" -d extracted ;;
      *gzip\ compressed*) tar -xzf "$1" -C extracted ;;
      *POSIX\ tar\ archive*) tar -xf "$1" -C extracted ;;
      *) echo "(unknown type for $1: $(file -b "$1"))" ;; esac; }
  for f in dataset_part_1 dataset_part_2; do extract $f; done
  found=0
  while IFS= read -r -d '' d; do
    if [ -d "$d/train/images" ] && [ -d "$d/test/images" ]; then cp -a "$d" "$DATA2D"; found=1; break; fi
  done < <(find extracted -type d -print0)
  if [ "$found" = 0 ]; then   # fallback: the two files may be parts of ONE archive
    cat dataset_part_1 dataset_part_2 > dataset_combined; extract dataset_combined
    while IFS= read -r -d '' d; do
      if [ -d "$d/train/images" ] && [ -d "$d/test/images" ]; then cp -a "$d" "$DATA2D"; found=1; break; fi
    done < <(find extracted -type d -print0)
  fi
  [ "$found" = 1 ] || { echo "ERROR: could not find train/images + test/images after extraction"; exit 1; }
fi
for sp in train test; do
  echo "$sp: $(ls "$DATA2D/$sp/images" | wc -l) images, $(ls "$DATA2D/$sp/annotations" | wc -l) annotations"
done
echo "2D DATASET OK -> $DATA2D"
mkdir -p "$REPO/pretrain"
