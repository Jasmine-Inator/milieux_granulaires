@echo
winget install python3.10
pip install virtualenv
virtualenv --python C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe venv
cd venv/Scripts
call activate.bat
cd../..
pip install "cython<3.0.0" Wheel
pip install "pyyaml==5.4.1" --no-build-isolation
pip install -r requirements.txt
pip install "protobuf<3.20"
cd venv/Lib/site-packages/object_detection
for /f "tokens=*" %%G1 in ('cd') do set dpath=%%G1
cd../../../../../../Tensorflow/models/research/object_detection
for /f "tokens=*" %%G2 in ('cd') do set opath=%%G2
xcopy %opath% %dpath% /s /e
cd../../official
for /f "tokens=*" %%G3 in ('cd') do set opath1=%%G3
cd %dpath%
mkdir official
cd official
for /f "tokens=*" %%G4 in ('cd') do set dpath1=%%G4
xcopy %opath1% %dpath1% /s /e
cmd /k