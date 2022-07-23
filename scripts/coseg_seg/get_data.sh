#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
tar -xzvf coseg.tar.gz && rm coseg.tar.gz