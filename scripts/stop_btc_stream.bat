@echo off
echo Stopping BTC price data stream...

:: Find and kill the Python process running btc_price_stream.py
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /v ^| findstr "btc_price_stream"') do (
    echo Killing process %%a
    taskkill /PID %%a /F
)

echo BTC price stream stopped.