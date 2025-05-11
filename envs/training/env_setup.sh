#!/bin/bash
sudo apt-get install python3.10
pip install virtualenv
virtualenv --python /c/Users/$(whoami)/AppData/Local/Programs/Python/Python310/python.exe
cd venv/Scripts
source ./activate
cd../..
pip install "cython<3.0.0" Wheel
pip install "pyyaml==5.4.1" --no-build-isolation
pip install -r requirements.txt
pip install "protobuf<3.20"
cd venv/Lib/site-packages/object_detection
dpath=$(pwd)
cd../../../../../../Tensorflow/models/research/object_detection
opath=$(pwd)
cp -a /$opath/. /$dpath/
cd../../official
opath1==$(pwd)
cd %dpath%
cd..
mkdir official
cd official
dpath1==$(pwd)
cp -a /$opath1/. /$dpath1/