@echo off

rem PyTorch profile: set LETS_TORCH=2.14 runs the backend in sd_scripts\venv_t214 (installer.py --torch 2.14);
rem with only venv_t214 installed it is used anyway
set "_LT=%LETS_TORCH%"
if defined _LT set "_LT=%_LT: =%"
set _T214=
for %%v in (2.14 2.14.0 214) do if /I "%_LT%"=="%%v" set _T214=1
if not exist sd_scripts\venv\Scripts\activate.bat if exist sd_scripts\venv_t214\Scripts\activate.bat set _T214=1
if defined _T214 (call sd_scripts\venv_t214\Scripts\activate) else (call sd_scripts\venv\Scripts\activate)
python main.py
pause
