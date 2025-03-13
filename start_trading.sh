#!/bin/bash
# Start the Know-Defeat trading system with proper environment settings

# Set the current directory to the project root
cd "$(dirname "$0")"

# Add the current directory to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Starting Know-Defeat trading system..."
echo "Project root: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# First, check if algorithm imports work
echo "Testing algorithm imports..."
python3 scripts/test_algorithm_imports.py

# Ask to continue
read -p "Do you want to continue and start the trading system? (y/n) " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
  echo "Exiting without starting the trading system."
  exit 0
fi

# Start the IB controller for tick data collection
echo "Starting IB controller for tick data collection..."
python3 src/ib_controller_simple.py &
IB_PID=$!

# Wait for IB controller to initialize
echo "Waiting 10 seconds for IB controller to initialize..."
sleep 10

# Start the trading bots with the correct algorithm directory
echo "Starting trading bots..."
python3 src/run_bots.py --algo_dir src/bots

# When the trading bots exit, also stop the IB controller
echo "Stopping IB controller..."
kill $IB_PID