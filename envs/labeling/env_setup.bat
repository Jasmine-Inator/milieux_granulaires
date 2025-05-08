@echo
winget install python3.9
virtualenv --python C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe
cd venv/Scripts
call activate.bat
pip install labelImg
