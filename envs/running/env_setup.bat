@echo
winget install python3.11
virtualenv --python C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
cd venv/Scripts
call activate.bat
cd../..
pip install "cython<3.0.0" Wheel
pip install "pyyaml==5.4.1" --no-build-isolation
pip install -r requirements.txt
pip install "protobuf<3.20"
cd venv/Lib/site-packages/object_detection
for /f "tokens=*" %%G in ('cd') do set dpath=%%G
cd../../../../../../Tensorflow/models/research/object_detection
for /f "tokens=*" %%G in ('cd') do set opath=%%G
xcopy %opath% %dpath% /h /i /c /k /e /r /y
cd../../official
for /f "tokens=*" %%G in ('cd') do set opath1=%%G
cd %dpath%
mkdir official
cd official
for /f "tokens=*" %%G in ('cd') do set dpath1=%%G
xcopy %opath1% %dpath1% /h /i /c /k /e /r /y