#!/bin/bash

# Stop the BTC price stream service
echo "Stopping BTC price stream..."

# Check for PID file
if [ -f "btc_stream.pid" ]; then
    PID=$(cat btc_stream.pid)
    
    # Verify process is running
    if ps -p $PID > /dev/null; then
        echo "Stopping BTC price stream (PID: $PID)"
        kill $PID
        rm btc_stream.pid
        echo "BTC price stream stopped"
    else
        echo "BTC price stream is not running (stale PID file)"
        rm btc_stream.pid
    fi
else
    # Try to find the process
    PID=$(pgrep -f "python3 scripts/btc_price_stream.py")
    if [ ! -z "$PID" ]; then
        echo "Stopping BTC price stream (PID: $PID)"
        kill $PID
        echo "BTC price stream stopped"
    else
        echo "BTC price stream is not running"
    fi
fi