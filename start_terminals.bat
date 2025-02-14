#!/bin/bash

@echo off
REM ================================================
REM Step 1: Start the Database
echo Starting database...
call conda activate Autogen
pg_ctl -D "C:\Users\clayb\postgres_data" start
echo Waiting 10 seconds for the database to become ready...
timeout /t 10 /nobreak

REM (Optional) Launch psql console in its own window
echo Launching psql console...
start "psql Console" cmd /k "conda activate Autogen && psql -U clayb tick_data"

REM ================================================
REM Step 2: Start the IB Controller in its own window
echo Starting IB Controller...
start "IB Controller" cmd /k "conda activate Autogen && python src/ib_controller.py"
echo Waiting 5 seconds for the IB Controller to initialize...
timeout /t 5 /nobreak

REM ================================================
REM Step 3: Start all Trading Bots in the same window
echo Starting trading bots (all in one window)...
REM Using start /B to launch each process in the background of the current window.
start /B "" cmd /c "conda activate Autogen && python src/bots/COIN_long_bot.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/COIN_short_bot.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/TSLA_long_bot.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/TSLA_short_bot.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/COIN_long_bot2.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/COIN_short_bot2.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/TSLA_long_bot2.py"
start /B "" cmd /c "conda activate Autogen && python src/bots/TSLA_short_bot2.py"

echo All processes started.
pause                   