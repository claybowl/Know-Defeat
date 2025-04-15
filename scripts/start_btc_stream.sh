#!/bin/bash

# Start the BTC price stream service
echo "Starting BTC price stream..."

# Check if process is already running
PID=$(pgrep -f "python3 scripts/btc_price_stream.py")
if [ ! -z "$PID" ]; then
    echo "BTC price stream is already running with PID $PID"
    exit 1
fi

# Start the service in the background
nohup python3 scripts/btc_price_stream.py > btc_stream.log 2>&1 &
PID=$!

echo "BTC price stream started with PID $PID"
echo $PID > btc_stream.pid