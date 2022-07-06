#!/bin/bash

file1="./models/model.pth.tar"

if [ ! -f "$file1" ] ; then
  wget https://owncloud.fraunhofer.de/index.php/s/lkPsFHcmXJqnLuu/download -P ./models/naturident_example.zip
  unzip ./models/naturident_example.zip 'models/*' -d './models'
  rm ./models/naturident_example.zip

fi

file2="./Datasets/pallet-block-32965"

if [ ! -d "$file2" ] ; then
  wget https://zenodo.org/record/6358607/files/pallet-block-32965.rar?download=1 -P ./Datasets
fi
