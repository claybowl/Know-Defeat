@echo off
echo Querying latest BTC prices...

:: Default to 10 records unless specified
set COUNT=10
if not "%1"=="" set COUNT=%1

python scripts\btc_price_stream.py --query --count %COUNT%

echo Query complete.