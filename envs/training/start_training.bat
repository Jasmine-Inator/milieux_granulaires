@echo 
cd venv/Scripts
call activate.bat
cd ../..
set /p shape= "Which shape are you training for (use _ isntead of spaces)?"
cd../..
cd Tensorflow/Workspace/training_demo_%shape%
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config