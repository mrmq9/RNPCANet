#!/usr/bin/env bash

DATA=$1

if [ $DATA = "coil-100" ]; then
  wget http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
unzip coil-100.zip
rm coil-100.zip
elif [ $DATA = "coil-20" ]; then
  wget http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip
unzip coil-20-proc.zip
rm coil-20-proc.zip
mv coil-20-proc coil-20
else
  echo "Unknown dataset"
fi
