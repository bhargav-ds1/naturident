#!/bin/bash

file1="./models/model.pth.tar"

if [ ! -f "$file1" ] ; then
  wget -O ./models/naturident_example.zip https://owncloud.fraunhofer.de/index.php/s/lkPsFHcmXJqnLuu/download
  pwd
  ls
  7zip ./models/naturident_example.zip 'models/*' -d './'
  rm ./models/naturident_example.zip

fi

file2="./Datasets/pallet-block-32965"

if [ ! -d "$file2" ] ; then
  wget -O ./Datasets/pallet-block-32965.rar https://zenodo.org/record/6358607/files/pallet-block-32965.rar?download=1
  7zip x pallet-block-32965.rar
  rm pallet-block-32965.rar
fi
