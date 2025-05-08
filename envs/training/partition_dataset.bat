@echo 
cd venv/Scripts
call activate.bat
set /p shape= "Which shape are you partitionning the dataset for (use _ instead of spaces)?"
set /p ratio= "Which ratio of images do you want to use for training (ie. 0.1 means 90 percent of images are put for training and 10 percent are put for testing)?"
cd../..
cd../..
cd Tensorflow/workspace/training_demo_%shape%/images
for /f "tokens=*" %%G in ('cd') do set impath=%%G
cd ../../..
cd scripts/preprocessing
python partition_dataset.py -x -i %impath% -r %ratio%
