#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR

tar -xzvf cubes.tar.gz && rm cubes.tar.gz
echo "downloaded the data and put it in: " $DATADIR