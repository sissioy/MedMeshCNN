#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/coseg_aliens \
--name coseg_aliens \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 12 \
--export_folder meshes \

python test.py --dataroot datasets/coseg_vases --name coseg_vases --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 --ninput_edges 1500 --pool_res 1140 750 400 --num_threads 0 --resblocks 3 --batch_size 12 --lr 0.001 --num_aug 20 --slide_verts 0.2 --gpu_ids 0 --verbose_plot --weighted_loss 0.125 0.125 0.125 0.125
python test.py --dataroot datasets/coseg_vases --name coseg_vases --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 --ninput_edges 1500 --pool_res 1140 750 400 --resblocks 3 --batch_size 12 --export_folder meshes 