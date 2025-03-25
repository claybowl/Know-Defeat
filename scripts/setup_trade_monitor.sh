#!/bin/bash
# Setup script for Trade Monitor

echo "Installing required dependencies for Trade Monitor..."
pip install tabulate

echo "Setup complete! You can now run the trade monitor with:"
echo "python scripts/trade_monitor.py"