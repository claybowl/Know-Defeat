#!/bin/bash

# Know-Defeat Trading System Shutdown Script

echo "Stopping Know-Defeat Trading System background processes..."

# Define Project Directory (needed for stop_btc_stream.sh)
PROJECT_DIR="/c/Users/clayb/Desktop/CurveAI/Know-Defeat"

# Function to activate conda environment (might be needed for some kill commands or scripts)
activate_env() {
    echo "Activating Conda environment: Autogen"
    source C:/Users/clayb/anaconda3/etc/profile.d/conda.sh # Adjust path if necessary
    conda activate Autogen
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to activate Conda environment. Some stop commands might fail."
    fi
}

# Function to kill processes based on a pattern
kill_process() {
    local pattern="$1"
    local description="$2"

    echo "Attempting to stop $description ($pattern)..."
    # Use pkill -f to match the full command line
    if pgrep -f "$pattern" > /dev/null; then
        pkill -f -TERM "$pattern" # Send TERM signal first
        sleep 2 # Give processes time to shut down gracefully
        if pgrep -f "$pattern" > /dev/null; then
            echo "$description did not stop gracefully, sending KILL signal..."
            pkill -f -KILL "$pattern" # Force kill if still running
            sleep 1
        fi

        if ! pgrep -f "$pattern" > /dev/null; then
            echo "$description stopped successfully."
        else
            echo "Warning: Failed to stop $description completely."
        fi
    else
        echo "$description ($pattern) not found or already stopped."
    fi
}

# --- Step 1: Activate Environment ---
activate_env

# --- Step 2: Navigate to Project Directory ---
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to $PROJECT_DIR. Cannot run stop_btc_stream.sh. Exiting."
    exit 1
fi

# --- Step 3: Stop Specific Scripts ---

# Stop Monitoring Systems
kill_process "scripts/trade_monitor.py" "General Trade Monitor"
kill_process "scripts/monitor_pending_trades.py" "Pending Trade Monitor"

# Stop BTC Price Stream (using its dedicated script)
echo "Attempting to stop BTC price stream using ./scripts/stop_btc_stream.sh..."
if [ -f "./scripts/stop_btc_stream.sh" ]; then
    ./scripts/stop_btc_stream.sh
    if [ $? -ne 0 ]; then
        echo "Warning: ./scripts/stop_btc_stream.sh reported an error."
    else
        echo "BTC price stream stop script executed."
    fi
else
    echo "Warning: ./scripts/stop_btc_stream.sh not found."
fi

# Stop Notifications Listener
# Be specific to avoid killing the setup command if it was run manually
kill_process "src.db.notifications_setup --listen" "Notification Listener"

# Stop Front-End Servers
kill_process "src.websocket_server" "WebSocket Server"
kill_process "uvicorn src.main:app" "Uvicorn/FastAPI Server"

# Stop Core Trading System Components
kill_process "src/run_bots.py --algo_dir src/bots" "Bot Runner"
kill_process "src/ib_controller_simple.py" "IB Controller"

# --- Step 4: Final Message ---
echo "-----------------------------------------------------------"
echo "Know-Defeat System Shutdown Script Completed."
echo "Attempted to stop all background processes launched by start_system.sh."
echo "Please verify manually if any processes remain running if needed (use 'ps aux | grep python' or task manager)."
echo "PostgreSQL database was NOT stopped by this script."
echo "-----------------------------------------------------------"

exit 0 