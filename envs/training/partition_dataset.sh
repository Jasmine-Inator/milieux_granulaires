#!/bin/bash
cd venv/Scripts
source ./activate
read -p shape= "Which shape are you creating records for (use _ instead of spaces)?" shape
read -p "Which ratio of images do you want to use for training (ie. 0.1 means 90 percent of images are put for training and 10 percent are put for testing)?" ratio
cd../..
cd../..
cd Tensorflow/workspace/training_demo_$shape/images
impath=$(pwd)
cd ../../..
cd scripts/preprocessing
python partition_dataset.py -x -i $impath -r $ratio
