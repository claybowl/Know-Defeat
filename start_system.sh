#!/bin/bash

# Know-Defeat Trading System Startup Script

echo "Starting Know-Defeat Trading System..."

# Define Project Directory
PROJECT_DIR="/c/Users/clayb/Desktop/CurveAI/Know-Defeat"
POSTGRES_DATA_DIR="C:/Users/clayb/postgres_data"

# Function to activate conda environment
activate_env() {
    echo "Activating Conda environment: Autogen"
    source C:/Users/clayb/anaconda3/etc/profile.d/conda.sh # Adjust path if necessary
    conda activate Autogen
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate Conda environment. Exiting."
        exit 1
    fi
}

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" > /dev/null
}

# --- Step 1: Activate Environment ---
activate_env

# --- Step 2: Navigate to Project Directory ---
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to $PROJECT_DIR. Exiting."
    exit 1
fi

# --- Step 3: Start PostgreSQL Database ---
echo "Checking PostgreSQL status..."
if pg_isready -q -h localhost -p 5432 -U clayb; then
    echo "PostgreSQL is already running."
else
    echo "Starting PostgreSQL database..."
    pg_ctl -D "$POSTGRES_DATA_DIR" start
    sleep 5 # Give the server time to start
    if ! pg_isready -q -h localhost -p 5432 -U clayb; then
        echo "Error: Failed to start PostgreSQL. Please check logs. Exiting."
        exit 1
    fi
    echo "PostgreSQL started successfully."
fi

# --- Step 4: Run Database Health Check ---
echo "Running database health check..."
python scripts/db_health_check.py
if [ $? -ne 0 ]; then
    echo "Warning: Database health check failed or reported issues. Continuing..."
    # Decide if you want to exit here based on severity
    # exit 1
fi

# --- Step 5: Start Interactive Brokers Gateway (Manual Step) ---
echo "-----------------------------------------------------------"
echo "MANUAL STEP REQUIRED:"
echo "Please ensure Interactive Brokers Gateway is running and logged in."
echo "Verify it is listening on port 4002."
read -p "Press Enter to continue once IB Gateway is ready..."
echo "-----------------------------------------------------------"

# Check IB Gateway connectivity (optional basic check)
# nc -z 127.0.0.1 4002
# if [ $? -ne 0 ]; then
#     echo "Warning: Cannot connect to IB Gateway on 127.0.0.1:4002. Please verify it's running."
# fi

# --- Step 6: Register/Verify Bots ---
echo "Registering/Verifying bots..."
bash register_bots.sh
if [ $? -ne 0 ]; then
    echo "Warning: Bot registration script failed. Continuing..."
fi

# --- Step 7: Start Core Trading System Components ---
echo "Starting IB Controller..."
python src/ib_controller_simple.py &
IB_CONTROLLER_PID=$!
echo "IB Controller started with PID $IB_CONTROLLER_PID"
sleep 5 # Allow controller time to initialize

echo "Starting Bot Runner..."
# Check if run_bots is already running to avoid duplicates if script is rerun
if ! is_process_running "src/run_bots.py"; then
    python src/run_bots.py --algo_dir src/bots &
    RUN_BOTS_PID=$!
    echo "Bot Runner started with PID $RUN_BOTS_PID"
else
    echo "Bot Runner (run_bots.py) appears to be already running."
fi


# --- Step 8: Start Front-End Servers ---
echo "Starting FastAPI server (uvicorn)..."
# Check if uvicorn is already running for this app
if ! is_process_running "uvicorn src.main:app"; then
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload &
    UVICORN_PID=$!
    echo "Uvicorn started with PID $UVICORN_PID"
else
     echo "Uvicorn (src.main:app) appears to be already running."
fi

echo "Starting WebSocket server..."
if ! is_process_running "src.websocket_server"; then
    python -m src.websocket_server &
    WEBSOCKET_PID=$!
    echo "WebSocket server started with PID $WEBSOCKET_PID"
else
    echo "WebSocket server (src.websocket_server) appears to be already running."
fi


# --- Step 9: Setup and Start Notifications ---
echo "Setting up database notifications..."
python -m src.db.notifications_setup --setup
if [ $? -ne 0 ]; then
    echo "Warning: Notification setup failed. Continuing..."
fi

echo "Starting notification listener..."
if ! is_process_running "src.db.notifications_setup --listen"; then
    python -m src.db.notifications_setup --listen &
    NOTIF_LISTEN_PID=$!
    echo "Notification listener started with PID $NOTIF_LISTEN_PID"
else
    echo "Notification listener (notifications_setup --listen) appears to be already running."
fi


# --- Step 10: Start BTC Price Stream ---
echo "Starting BTC price stream..."
# Assuming start_btc_stream.sh handles PID checking internally or runs in background
./scripts/start_btc_stream.sh
if [ $? -ne 0 ]; then
    echo "Warning: Failed to start BTC price stream. Check logs/scripts/start_btc_stream.sh."
fi

# --- Step 11: Start Monitoring Systems ---
echo "Starting Pending Trade Monitor..."
if ! is_process_running "scripts/monitor_pending_trades.py"; then
    python scripts/monitor_pending_trades.py &
    PENDING_MONITOR_PID=$!
    echo "Pending Trade Monitor started with PID $PENDING_MONITOR_PID"
else
    echo "Pending Trade Monitor (monitor_pending_trades.py) appears to be already running."
fi

echo "Starting General Trade Monitor..."
if ! is_process_running "scripts/trade_monitor.py"; then
    python scripts/trade_monitor.py &
    TRADE_MONITOR_PID=$!
    echo "General Trade Monitor started with PID $TRADE_MONITOR_PID"
else
    echo "General Trade Monitor (trade_monitor.py) appears to be already running."
fi

# --- Step 12: Final Instructions ---
echo "-----------------------------------------------------------"
echo "Know-Defeat System Startup Initiated."
echo "Core components and monitors are starting in the background."
echo "Check logs for detailed status:"
echo "  - tail -f trading_system.log"
echo "  - tail -f logs/trade_logs/latest_trades.log"
echo "  - Check individual script logs (e.g., logs/ directory)"
echo ""
echo "Optional Steps:"
echo " - Start Streamlit UI: streamlit run user_interface/src/streamlit_app2.py"
echo " - Monitor DB directly: psql -U clayb -d tick_data"
echo " - Check background PIDs:"
echo "   - IB Controller: $IB_CONTROLLER_PID"
echo "   - Bot Runner: $RUN_BOTS_PID"
echo "   - Uvicorn: $UVICORN_PID"
echo "   - WebSocket: $WEBSOCKET_PID"
echo "   - Notif Listener: $NOTIF_LISTEN_PID"
echo "   - Pending Monitor: $PENDING_MONITOR_PID"
echo "   - Trade Monitor: $TRADE_MONITOR_PID"
echo "   - BTC Stream: (Check scripts/start_btc_stream.sh or btc_stream.pid)"
echo ""
echo "To stop background processes, use 'kill <PID>' or relevant stop scripts (like stop_btc_stream.sh)."
echo "-----------------------------------------------------------"

exit 0 