@echo
cd venv/Scripts
call activate.bat
set /p shape= "Which shape are you creating records for (use _ instead of spaces)?"
cd../..
cd../..
cd Tensorflow/workspace/training_demo_%shape%/images
for /f "tokens=*" %%G in ('cd') do set impath=%%G
cd ..
cd annotations
for /f "tokens=*" %%G in ('cd') do set anpath=%%G
cd ../../..
cd scripts/preprocessing
python generate_tfrecord.py -x %impath%/train -l %anpath%/label_map.pbtxt -o %anpath%/train.record
python generate_tfrecord.py -x %impath%/test -l %anpath%/label_map.pbtxt -o %anpath%/test.record