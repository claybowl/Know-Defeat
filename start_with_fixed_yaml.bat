@echo off
REM Start the Know-Defeat trading system with fixed YAML paths

REM Set the current directory to the project root
cd /d "%~dp0"

REM First fix the YAML paths
echo Fixing algorithm paths in YAML files...
python fix_yaml_paths.py

REM Run the startup script
echo Running main startup script...
call start_trading.bat