#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

# get data
echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
tar -xzvf human_seg.tar.gz && rm human_seg.tar.gz