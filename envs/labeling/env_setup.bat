@echo
winget install python3.9
pip install virtualenv
virtualenv --python C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe venv
cd venv/Scripts
call activate.bat
pip install labelImg
deactivate
cmd /k