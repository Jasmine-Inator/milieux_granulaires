#!/bin/bash
cd venv/Scripts
source ./activate
read -p shape= "Which shape are you creating records for (use _ instead of spaces)?" shape
cd../..
cd../..
cd Tensorflow/workspace/training_demo_$shape/images
impath=$(pwd)
cd ..
cd annotations
anpath=$(pwd)
cd ../../..
cd scripts/preprocessing
python generate_tfrecord.py -x $impath/train -l $anpath/label_map.pbtxt -o $anpath/train.record
python generate_tfrecord.py -x $impath/test -l $anpath/label_map.pbtxt -o $anpath/test.record