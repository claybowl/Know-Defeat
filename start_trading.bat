@echo off
REM Start the Know-Defeat trading system with proper environment settings

REM Set the current directory to the project root
cd /d "%~dp0"

REM Add the current directory to PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

echo Starting Know-Defeat trading system...
echo Project root: %CD%
echo PYTHONPATH: %PYTHONPATH%

REM First, check if algorithm imports work
echo Testing algorithm imports...
python scripts/test_algorithm_imports.py

REM Additional test for module import paths
echo Testing module import paths...
python scripts/test_module_paths.py

REM Ask to continue
set /p answer=Do you want to continue and start the trading system? (y/n) 
if /i "%answer%" neq "y" (
  echo Exiting without starting the trading system.
  exit /b
)

REM Start the IB controller for tick data collection in a new window
echo Starting IB controller for tick data collection...
start "IB Controller" cmd /c python src/ib_controller_simple.py

REM Wait for IB controller to initialize
echo Waiting 10 seconds for IB controller to initialize...
timeout /t 10

REM Start the trading bots with the correct algorithm directory
echo Starting trading bots...
python src/run_bots.py --algo_dir src/bots

echo Trading system has stopped.