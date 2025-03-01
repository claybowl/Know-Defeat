
@echo off
call "C:\Users\clayb/Anaconda3/Scripts/conda.exe" activate Autogen
echo Starting IB Controller...

REM Set PostgreSQL connection parameters to avoid connection issues
set PGCONNECT_TIMEOUT=10
set PGPOOL_MIN_CONN=2
set PGPOOL_MAX_CONN=5

python src/ib_controller.py
