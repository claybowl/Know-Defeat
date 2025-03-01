
@echo off
call "C:\Users\clayb/Anaconda3/Scripts/conda.exe" activate Autogen
echo Starting COIN_long with ID 1...
REM Add a random delay to avoid database connection contention
timeout /t 2 /nobreak > nul

REM Run the bot with proper connection pool settings
set PGCONNECT_TIMEOUT=10
set PGPOOL_MIN_CONN=1
set PGPOOL_MAX_CONN=2
python src\bots\COIN_long_bot.py --bot_id 1
