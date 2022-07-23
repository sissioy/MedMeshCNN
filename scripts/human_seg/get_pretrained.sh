#!/usr/bin/env bash

CHECKPOINT='checkpoints/human_seg'
mkdir -p $CHECKPOINT

tar -xzvf human_seg_wts.tar.gz && rm human_seg_wts.tar.gz
mv latest_net.pth $CHECKPOINT
echo "downloaded pretrained weights to" $CHECKPOINT
