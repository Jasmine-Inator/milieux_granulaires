#!/bin/bash
sudo apt-get install python3.9
pip install virtualenv
virtualenv --python /c/Users/$(whoami)/AppData/Local/Programs/Python/Python39/python.exe
cd venv/Scripts
source ./activate
pip install labelImg
