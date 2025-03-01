
@echo off
call "C:\Users\clayb/Anaconda3/Scripts/conda.exe" activate Autogen
echo Starting COIN_short2 with ID 4...
REM Add a random delay to avoid database connection contention
timeout /t 8 /nobreak > nul

REM Run the bot with proper connection pool settings
set PGCONNECT_TIMEOUT=10
set PGPOOL_MIN_CONN=1
set PGPOOL_MAX_CONN=2
python src\bots\COIN_short_bot2.py --bot_id 4
