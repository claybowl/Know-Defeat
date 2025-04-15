#!/bin/bash

# Query the latest BTC prices from the database
echo "Querying latest BTC prices..."

# Default to 10 records unless specified
COUNT=${1:-10}

python3 scripts/btc_price_stream.py --query --count $COUNT

echo "Query complete."