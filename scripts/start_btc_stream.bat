@echo off
echo Starting BTC price data stream...

:: Run in background using start command
start /B python scripts\btc_price_stream.py > btc_stream.log 2>&1

echo BTC price stream started. Check btc_stream.log for output.