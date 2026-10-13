DATA=/path/to/NC-release-data-checked
python ./train.py --dataset 3D-CBCT-Tooth --model TMamba3D --dimension 3d \
    --dataset_path $DATA --gpu 0 \
    --pretrain_weight ./runs_ssl/unimiss3d/unimiss_encoder_teacher.pth \
    --scaling_version TINY --epoch 20
