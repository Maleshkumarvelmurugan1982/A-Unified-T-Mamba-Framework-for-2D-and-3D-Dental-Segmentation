DATA=/path/to/NC-release-data-checked
python ./test.py --dataset 3D-CBCT-Tooth --model TMamba3D --dimension 3d --scaling_version TINY \
    --dataset_path $DATA --gpu 0 --pretrain_weight ./pretrain/best_TMamba3D.pth
