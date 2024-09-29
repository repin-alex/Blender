@echo off
REM Устанавливаем путь к Python, который находится в папке проекта
SET PYTHON_PATH=%~dp0Python312\python.exe

REM Обновляем pip для текущего проекта
%PYTHON_PATH% -m pip install --upgrade pip

REM Устанавливаем зависимости из requirements.txt
%PYTHON_PATH% -m pip install -r %~dp0requirements.txt

REM Запускаем основной скрипт (например, main.py или api.py)
%PYTHON_PATH% %~dp0api.py

%PYTHON_PATH% -m uvicorn api:app --reload

REM Заглушка, чтобы терминал не закрывался автоматически
echo.
echo Нажмите любую клавишу, чтобы закрыть окно...
pause >nul