import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import os
import time
import sys
import logging
import pickle
import psycopg2
import asyncio
import asyncpg
from datetime import datetime, timedelta
from collections import defaultdict
from plotly.subplots import make_subplots
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from weights_management_ui import WeightsManagementUI
from trade_manager import TradeManager
from ai_weight_adjuster import AIWeightAdjuster

# Define helper functions for the dashboard
def show_dashboard():
    """Display the main system dashboard."""
    st.title("System Dashboard")
    st.info("The main dashboard content goes here. Select 'Fund Allocation Dashboard' from the navigation to see the new features.")

def show_metrics():
    """Display bot metrics."""
    st.title("Bot Metrics")
    st.info("The bot metrics content would go here. Select 'Fund Allocation Dashboard' from the navigation to see the new features.")

def show_raw_metrics():
    """Display raw bot metrics data."""
    st.title("Raw Bot Metrics")
    st.info("Raw bot metrics would be displayed here. Select 'Fund Allocation Dashboard' from the navigation to see the new features.")

def show_raw_rankings():
    """Display raw bot rankings data."""
    st.title("Raw Bot Rankings")
    st.info("Raw bot rankings would be displayed here. Select 'Fund Allocation Dashboard' from the navigation to see the new features.")

def show_db_check():
    """Display database check results."""
    st.title("Database Check")
    
    # Run check in background
    with st.spinner("Checking database schema..."):
        db_status = asyncio.run(check_database_schema())
    
    st.success("Database schema check complete")

# Forward declaration for functions used before defined
async def load_logs_from_db(bot_name=None, limit=1000):
    pass

# Set page config with a cool style
st.set_page_config(
    page_title="Know Defeat",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Initialize session state variables at the start of the app
if 'bot_processes' not in st.session_state:
    st.session_state.bot_processes = {
        'COIN_long': None,
        'COIN_short': None,
        'TSLA_long': None,
        'TSLA_short': None,
        'COIN_long2': None,
        'COIN_short2': None,
        'TSLA_long2': None,
        'TSLA_short2': None
    }

if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = {
        'ib_controller': [],
        'COIN_long': [],
        'COIN_short': [],
        'TSLA_long': [],
        'TSLA_short': [],
        'COIN_long2': [],
        'COIN_short2': [],
        'TSLA_long2': [],
        'TSLA_short2': []
    }

if 'ib_controller_process' not in st.session_state:
    st.session_state.ib_controller_process = None

if 'risk_per_trade' not in st.session_state:
    st.session_state.risk_per_trade = 1.0

# Dashboard state management
if 'logs' not in st.session_state:
    st.session_state.logs = []

st.title("Know Defeat Trading System by Curve Ai Solutions")


################################################################################################################################

# Add Dev Blog section immediately after the title and before the tabs
with st.expander("📝 Development Blog", expanded=True):
    st.markdown("""
    ## Development Updates
    
    This space contains the latest development updates for partners. Check back regularly for new information on features, improvements, and upcoming changes.
    
    ### Latest Updates

    #### March 7, 2025 - Interactive Brokers Integration
    Added comprehensive IB account monitoring to the trading dashboard:
    - Real-time account summary showing balances, P&L, and leverage
    - Current positions tracking with performance metrics
    - Historical account value tracking and visualization
    - Automatic data storage in the database for long-term analysis
    
    #### March 5, 2025 - Top 10 Fund Allocation System
    Implemented a dynamic fund allocation system based on bot rankings:
    - Automatically allocates $2,000 to each bot in the top 10
    - Removes funding from bots that fall out of the top 10
    - Adds funding to bots that enter the top 10
    - Provides visualization of allocation changes over time
    - Includes a historical record of all allocation changes
    
    #### February 28, 2025 - Connection Pooling Optimization
    Improved database connection management:
    - Implemented connection pooling to reduce overhead
    - Added database status monitoring tools
    - Automated cleanup of idle connections
    - Enhanced error handling for database operations
    
    #### February 20, 2025 - Bot Ranking System Launch
    Launched the new bot ranking system:
    - Automatic evaluation of bots based on multiple performance metrics
    - Weight management for customizing the importance of different metrics
    - Historical tracking of bot rankings over time
    - Integration with the database for persistent storage
    
    _This area will be populated with development updates. Partners can refer to this section for the most recent changes and progress on the Know Defeat Trading System._
    
    ---
    
    *You can add new updates here in markdown format. Include dates, feature descriptions, bug fixes, and any other relevant development information.*
    """)

################################################################################################################################


# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

# Utility function to create a database connection pool with custom settings
async def create_db_pool(min_size=None, max_size=None, timeout=None, **kwargs):
    """Create a database connection pool with custom settings"""
    config = dict(DB_CONFIG)
    
    # Override settings if provided
    if min_size is not None:
        config['min_size'] = min_size
    if max_size is not None:
        config['max_size'] = max_size
    if timeout is not None:
        config['timeout'] = timeout
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return await asyncpg.create_pool(**config)

# Helper function to safely convert percentage strings to numeric values
def safe_pct_to_numeric(series):
    """
    Safely convert a pandas Series containing percentage strings or numeric values to numeric.
    
    Args:
        series: Pandas Series that might contain percentage strings or numeric values
        
    Returns:
        Pandas Series containing only numeric values
    """
    # First try to convert directly to numeric
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        pass
    
    # If the above fails, try string processing only if we have strings
    try:
        return series.str.rstrip('%').astype('float') / 100.0
    except Exception as e:
        # Return original if all else fails
        return series

# Helper function to safely format values that might be None
def safe_format(value, format_str):
    """
    Safely format a value that might be None.
    
    Args:
        value: The value to format
        format_str: The format string to use
        
    Returns:
        Formatted string or None if value is None
    """
    if value is None:
        return None
    return format_str.format(value)

def start_ib_controller():
    """Start the IB Controller process"""
    try:
        if not st.session_state.ib_controller_process:
            # Use the full path to conda and activate the Autogen environment
            conda_path = os.path.expanduser("~/Anaconda3/Scripts/conda.exe")
            
            # Create a batch script with improved database connection handling
            batch_script = """
@echo off
call "{conda_path}" activate Autogen
echo Starting IB Controller...

REM Set PostgreSQL connection parameters to avoid connection issues
set PGCONNECT_TIMEOUT=10
set PGPOOL_MIN_CONN=2
set PGPOOL_MAX_CONN=5

python src/ib_controller.py
""".format(conda_path=conda_path)
            
            # Write the batch script to a temporary file
            batch_file = os.path.join(os.getcwd(), "run_ib_controller.bat")
            with open(batch_file, "w") as f:
                f.write(batch_script)
            
            # Run the batch file
            process = subprocess.Popen(
                [batch_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                shell=True
            )
            st.session_state.ib_controller_process = process
            return True
        return False
    except Exception as e:
        st.error(f"Failed to start IB Controller: {e}")
        return False

def start_bot(bot_name):
    """Start a trading bot process"""
    try:
        if not st.session_state.bot_processes[bot_name]:
            # Map bot names to their correct bot_ids
            bot_id_mapping = {
                'COIN_long': 1,
                'COIN_short': 2,
                'COIN_long2': 3,
                'COIN_short2': 4,
                'TSLA_long': 5,
                'TSLA_short': 6,
                'TSLA_long2': 7,
                'TSLA_short2': 8
            }
            
            # Get the correct bot_id
            bot_id = bot_id_mapping.get(bot_name)
            if bot_id is None:
                st.error(f"Unknown bot name: {bot_name}")
                return False
                
            script_path = os.path.join('src', 'bots', f'{bot_name}_bot.py')
            # If the script filename differs (e.g., ..._bot2.py), you might need logic to pick the right filename
            if not os.path.exists(script_path):  
                # Fallback for "2" version naming
                script_path_2 = os.path.join('src', 'bots', f'{bot_name}.py')
                if os.path.exists(script_path_2):
                    script_path = script_path_2
                else:
                    # Try removing the "2" from the filename for version 2 bots
                    base_name = bot_name.replace('2', '')
                    script_path = os.path.join('src', 'bots', f'{base_name}_bot2.py')
                    if not os.path.exists(script_path):
                        st.error(f"Could not find script for bot: {bot_name}")
                        return False

            # Use the full path to conda and activate the Autogen environment
            conda_path = os.path.expanduser("~/Anaconda3/Scripts/conda.exe")
            
            # Create a batch script with improved connection handling
            batch_script = """
@echo off
call "{conda_path}" activate Autogen
echo Starting {bot_name} with ID {bot_id}...
REM Add a random delay to avoid database connection contention
timeout /t {random_delay} /nobreak > nul

REM Run the bot with proper connection pool settings
set PGCONNECT_TIMEOUT=10
set PGPOOL_MIN_CONN=1
set PGPOOL_MAX_CONN=2
python {script_path} --bot_id {bot_id}
""".format(
    conda_path=conda_path, 
    script_path=script_path, 
    bot_id=bot_id, 
    bot_name=bot_name,
    random_delay=bot_id*2  # Stagger start times based on bot_id
)
            
            # Write the batch script to a temporary file
            batch_file = os.path.join(os.getcwd(), f"run_{bot_name}.bat")
            with open(batch_file, "w") as f:
                f.write(batch_script)
            
            # Run the batch file
            process = subprocess.Popen(
                [batch_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                shell=True
            )
            st.session_state.bot_processes[bot_name] = process
            return True
        return False
    except Exception as e:
        st.error(f"Failed to start {bot_name} bot: {e}")
        return False

def stop_process(process_type, bot_name=None):
    """Stop a running process"""
    try:
        if bot_name:
            process = st.session_state.bot_processes[bot_name]
            if process:
                process.terminate()
                process.wait(timeout=5)
                st.session_state.bot_processes[bot_name] = None
                st.session_state.log_buffer[bot_name] = []
                
                # Clean up the batch file
                batch_file = os.path.join(os.getcwd(), f"run_{bot_name}.bat")
                if os.path.exists(batch_file):
                    try:
                        os.remove(batch_file)
                    except Exception as e:
                        st.warning(f"Could not remove batch file {batch_file}: {e}")
                
                # Add a small cleanup script to close any lingering connections
                cleanup_query = f"""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE application_name LIKE '%{bot_name}%' 
                AND pid <> pg_backend_pid();
                """
                try:
                    asyncio.run(_run_cleanup_query(cleanup_query))
                except Exception as e:
                    st.warning(f"Could not clean up database connections: {e}")
        else:
            process = st.session_state.ib_controller_process
            if process:
                process.terminate()
                process.wait(timeout=5)
                st.session_state.ib_controller_process = None
                st.session_state.log_buffer['ib_controller'] = []
                
                # Clean up the batch file
                batch_file = os.path.join(os.getcwd(), "run_ib_controller.bat")
                if os.path.exists(batch_file):
                    try:
                        os.remove(batch_file)
                    except Exception as e:
                        st.warning(f"Could not remove batch file {batch_file}: {e}")
                
                # Clean up IB controller database connections
                cleanup_query = """
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE application_name LIKE '%ib_controller%' 
                AND pid <> pg_backend_pid();
                """
                try:
                    asyncio.run(_run_cleanup_query(cleanup_query))
                except Exception as e:
                    st.warning(f"Could not clean up database connections: {e}")
    except Exception as e:
        st.error(f"Error stopping process: {e}")

async def _run_cleanup_query(query):
    """Run a cleanup query to terminate lingering database connections"""
    try:
        # Use our utility function to create a small connection pool
        pool = await create_db_pool(min_size=1, max_size=1)
        
        async with pool:
            await pool.execute(query)
    except Exception as e:
        logging.error(f"Error in cleanup query: {e}")
        raise

def update_logs():
    """Update log buffers for all running processes"""
    try:
        # Update IB Controller logs
        if st.session_state.ib_controller_process:
            while True:
                line = st.session_state.ib_controller_process.stdout.readline()
                if not line:
                    break
                # Format the log with timestamp if not present
                log_line = line.strip()
                if log_line and not (log_line.startswith("20") or log_line.startswith("202")):
                    log_line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {log_line}"
                st.session_state.log_buffer['ib_controller'].append(log_line)

        # Update bot logs
        for bot_name, process in st.session_state.bot_processes.items():
            if process:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    # Format the log with timestamp if not present
                    log_line = line.strip()
                    if log_line and not (log_line.startswith("20") or log_line.startswith("202")):
                        log_line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {log_line}"
                    st.session_state.log_buffer[bot_name].append(log_line)
    except Exception as e:
        st.error(f"Error updating logs: {e}")

def log_bot_event(bot_name, event_type, message):
    """
    Log a specific event for a bot
    
    Args:
        bot_name (str): Name of the bot
        event_type (str): Type of event (INFO, WARNING, ERROR, TRADE)
        message (str): Log message
    """
    if bot_name not in st.session_state.log_buffer:
        st.session_state.log_buffer[bot_name] = []
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"{timestamp} [{event_type}] {message}"
    st.session_state.log_buffer[bot_name].append(log_line)

# Add function to save logs to database
async def save_logs_to_db(bot_name=None):
    """
    Save logs to database for persistence
    
    Args:
        bot_name (str, optional): Name of the bot to save logs for. If None, save all logs.
    """
    try:
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Check if logs table exists, create if not
            await pool.execute("""
                CREATE TABLE IF NOT EXISTS bot_logs (
                    log_id SERIAL PRIMARY KEY,
                    bot_name VARCHAR(50) NOT NULL,
                    log_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Determine which logs to save
            if bot_name:
                logs_to_save = {bot_name: st.session_state.log_buffer.get(bot_name, [])}
            else:
                logs_to_save = st.session_state.log_buffer
            
            # Save logs to database
            for name, logs in logs_to_save.items():
                if logs:
                    # Only save the last 1000 log entries to avoid overwhelming the database
                    last_logs = logs[-1000:]
                    log_text = "\n".join(last_logs)
                    
                    await pool.execute("""
                        INSERT INTO bot_logs (bot_name, log_text)
                        VALUES ($1, $2);
                    """, name, log_text)
                    
            return True
    except Exception as e:
        st.error(f"Error saving logs to database: {e}")
        return False

# Create main sections using tabs
tab_controls, tab_logs, tab_tables, tab_trades, tab_params, tab_rankings, tab_account, tab_export = st.tabs([
    "Controls", "Logs", "Tables", "Trade Data", "Parameters", "Bot Rankings", "Account Info", "Data Export"
])

# Controls Section
with tab_controls:
    st.header("System Controls")

    # Database connection utilities
    st.subheader("Database Connection Utilities")
    
    if st.button("Check Database Status", help="Get detailed database status information"):
        try:
            async def check_db_status():
                try:
                    st.info("Connecting to database...")
                    # Use our utility function with a small connection pool
                    pool = await create_db_pool(min_size=1, max_size=1, timeout=5.0)
                    
                    async with pool:
                        # General database info
                        version = await pool.fetchval("SELECT version()")
                        st.write(f"**PostgreSQL version**: {version}")
                        
                        # Connection stats
                        conn_count = await pool.fetchval("""
                            SELECT count(*) FROM pg_stat_activity 
                            WHERE datname = $1
                        """, DB_CONFIG['database'])
                        
                        max_conn = await pool.fetchval("SHOW max_connections")
                        usage_pct = (conn_count / int(max_conn)) * 100
                        
                        # Create metrics display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Connections", f"{conn_count}")
                        with col2:
                            st.metric("Max Connections", f"{max_conn}")
                        with col3:
                            st.metric("Connection Usage", f"{usage_pct:.1f}%", 
                                      delta="High" if usage_pct > 80 else ("Moderate" if usage_pct > 50 else "Low"),
                                      delta_color="inverse")
                        
                        # Detailed connection information
                        st.subheader("Active Connections")
                        active_conns = await pool.fetch("""
                            SELECT 
                                pid,
                                application_name,
                                state,
                                query,
                                EXTRACT(EPOCH FROM (now() - query_start)) as query_duration,
                                EXTRACT(EPOCH FROM (now() - backend_start)) as connection_duration
                            FROM pg_stat_activity
                            WHERE datname = $1
                            ORDER BY state, query_duration DESC
                        """, DB_CONFIG['database'])
                        
                        if active_conns:
                            # Convert to DataFrame
                            conn_df = pd.DataFrame([dict(r) for r in active_conns])
                            
                            # Format the durations
                            conn_df['query_duration'] = conn_df['query_duration'].apply(
                                lambda x: f"{x:.1f}s" if x else "")
                            conn_df['connection_duration'] = conn_df['connection_duration'].apply(
                                lambda x: f"{x:.1f}s" if x else "")
                            
                            # Truncate long queries for display
                            conn_df['query'] = conn_df['query'].apply(
                                lambda x: (x[:150] + "...") if x and len(x) > 150 else x)
                            
                            st.dataframe(conn_df)
                        else:
                            st.info("No active connections found.")
                        
                        # Database size information
                        st.subheader("Database Size")
                        db_sizes = await pool.fetch("""
                            SELECT 
                                datname,
                                pg_size_pretty(pg_database_size(datname)) as size
                            FROM pg_database
                            ORDER BY pg_database_size(datname) DESC
                        """)
                        
                        if db_sizes:
                            st.dataframe(pd.DataFrame([dict(r) for r in db_sizes]))
                        
                        # Table sizes
                        st.subheader("Table Sizes")
                        table_sizes = await pool.fetch("""
                            SELECT 
                                table_name,
                                pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size,
                                pg_size_pretty(pg_relation_size(quote_ident(table_name))) as table_size,
                                pg_size_pretty(pg_total_relation_size(quote_ident(table_name)) - 
                                              pg_relation_size(quote_ident(table_name))) as index_size
                            FROM information_schema.tables
                            WHERE table_schema = 'public'
                            ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
                            LIMIT 10
                        """)
                        
                        if table_sizes:
                            st.dataframe(pd.DataFrame([dict(r) for r in table_sizes]))
                        else:
                            st.info("No tables found.")
                        
                except asyncpg.exceptions.PostgresError as e:
                    st.error(f"PostgreSQL error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    raise
            
            asyncio.run(check_db_status())
        except Exception as e:
            st.error(f"Error checking database status: {e}")
    
    col_db1, col_db2 = st.columns(2)
    
    with col_db1:
        if st.button("Check & Clean Database Connections"):
            try:
                async def check_and_clean_connections():
                    try:
                        st.info("Connecting to database...")
                        # Use our utility function to create a small connection pool
                        pool = await create_db_pool(min_size=1, max_size=1, timeout=5.0)
                        
                        async with pool:
                            # Check current connection count
                            conn_count = await pool.fetchval("""
                                SELECT count(*) FROM pg_stat_activity 
                                WHERE datname = $1
                            """, DB_CONFIG['database'])
                            
                            st.write(f"Current database connections: {conn_count}")
                            
                            # Check connection limit
                            max_conn = await pool.fetchval("SHOW max_connections")
                            st.write(f"Database max connections: {max_conn}")
                            
                            # Get connection usage percentage
                            usage_pct = (conn_count / int(max_conn)) * 100
                            st.write(f"Connection usage: {usage_pct:.1f}%")
                            
                            # Clean up idle connections
                            if conn_count > 50:  # If we have too many connections
                                st.warning(f"High connection count detected ({conn_count}). Cleaning up idle connections...")
                                cleaned = await pool.fetchval("""
                                    SELECT count(pg_terminate_backend(pid)) 
                                    FROM pg_stat_activity 
                                    WHERE datname = $1
                                    AND state = 'idle'
                                    AND pid <> pg_backend_pid()
                                """, DB_CONFIG['database'])
                                
                                st.success(f"Cleaned up {cleaned} idle database connections")
                            else:
                                st.success(f"Connection count is normal ({conn_count})")
                    except asyncpg.exceptions.PostgresError as e:
                        st.error(f"PostgreSQL error: {str(e)}")
                        st.info("Try using the EMERGENCY STOP button if the database is overloaded")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        raise
                
                asyncio.run(check_and_clean_connections())
            except Exception as e:
                st.error(f"Error checking database connections: {e}")
                st.error(f"Error type: {type(e).__name__}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
    with col_db2:
        if st.button("EMERGENCY STOP", type="primary", help="Stops all processes and cleans up all connections"):
            try:
                # Stop all bots
                st.info("Stopping all bots...")
                for bot_name in st.session_state.bot_processes.keys():
                    try:
                        stop_process('bots', bot_name)
                        st.info(f"Stopped {bot_name}")
                    except Exception as e:
                        st.error(f"Error stopping {bot_name}: {e}")
                
                # Stop IB controller
                st.info("Stopping IB controller...")
                try:
                    stop_process('ib_controller')
                    st.info("IB controller stopped")
                except Exception as e:
                    st.error(f"Error stopping IB controller: {e}")
                
                # Clean up all connections
                async def emergency_cleanup():
                    try:
                        st.info("Connecting to database for cleanup...")
                        # Use our utility function to create a small connection pool
                        pool = await create_db_pool(min_size=1, max_size=1, timeout=5.0)
                        
                        async with pool:
                            # Kill all connections except our own
                            st.info("Terminating all database connections...")
                            killed = await pool.fetchval("""
                                SELECT count(pg_terminate_backend(pid)) 
                                FROM pg_stat_activity 
                                WHERE datname = $1
                                AND pid <> pg_backend_pid()
                            """, DB_CONFIG['database'])
                            
                            return killed
                    except asyncpg.exceptions.PostgresError as e:
                        st.error(f"PostgreSQL error during emergency cleanup: {str(e)}")
                        return 0
                    except Exception as e:
                        st.error(f"Unexpected error during emergency cleanup: {str(e)}")
                        return 0
                
                try:
                    killed = asyncio.run(emergency_cleanup())
                    if killed > 0:
                        st.success(f"Emergency stop complete! Terminated {killed} database connections")
                    else:
                        st.warning("No database connections were terminated")
                except Exception as e:
                    st.error(f"Error during database connection cleanup: {e}")
                
                # Delete all batch files
                st.info("Cleaning up batch files...")
                try:
                    batch_files = [f for f in os.listdir(os.getcwd()) if f.startswith("run_") and f.endswith(".bat")]
                    for file in batch_files:
                        try:
                            os.remove(os.path.join(os.getcwd(), file))
                            st.info(f"Removed {file}")
                        except Exception as e:
                            st.error(f"Could not remove {file}: {e}")
                    
                    st.success(f"Cleaned up {len(batch_files)} batch files")
                except Exception as e:
                    st.error(f"Error cleaning up batch files: {e}")
                
                st.success("Emergency stop completed!")
                
            except Exception as e:
                st.error(f"Error during emergency stop: {e}")
                st.error(f"Error type: {type(e).__name__}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
    # IB Controller Controls
    st.subheader("IB Controller")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start IB Controller"):
            if start_ib_controller():
                st.success("IB Controller started successfully!")
    with col2:
        if st.button("Stop IB Controller"):
            stop_process('ib_controller')
            st.success("IB Controller stopped successfully!")

    # Trading Bots Controls
    st.subheader("Trading Bots")

    # First row of bots
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("COIN Long")
        if st.button("Start COIN Long"):
            if start_bot('COIN_long'):
                st.success("COIN Long bot started successfully!")
        if st.button("Stop COIN Long"):
            stop_process('bots', 'COIN_long')
            st.success("COIN Long bot stopped successfully!")

    with col2:
        st.write("COIN Short")
        if st.button("Start COIN Short"):
            if start_bot('COIN_short'):
                st.success("COIN Short bot started successfully!")
        if st.button("Stop COIN Short"):
            stop_process('bots', 'COIN_short')
            st.success("COIN Short bot stopped successfully!")

    with col3:
        st.write("TSLA Long")
        if st.button("Start TSLA Long"):
            if start_bot('TSLA_long'):
                st.success("TSLA Long bot started successfully!")
        if st.button("Stop TSLA Long"):
            stop_process('bots', 'TSLA_long')
            st.success("TSLA Long bot stopped successfully!")

    with col4:
        st.write("TSLA Short")
        if st.button("Start TSLA Short"):
            if start_bot('TSLA_short'):
                st.success("TSLA Short bot started successfully!")
        if st.button("Stop TSLA Short"):
            stop_process('bots', 'TSLA_short')
            st.success("TSLA Short bot stopped successfully!")

    # Second row of bots (the "2" versions)
    st.write("-----")
    st.write("Additional Bot Versions")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.write("COIN Long 2")
        if st.button("Start COIN Long 2"):
            if start_bot('COIN_long2'):
                st.success("COIN Long 2 bot started successfully!")
        if st.button("Stop COIN Long 2"):
            stop_process('bots', 'COIN_long2')
            st.success("COIN Long 2 bot stopped successfully!")

    with col6:
        st.write("COIN Short 2")
        if st.button("Start COIN Short 2"):
            if start_bot('COIN_short2'):
                st.success("COIN Short 2 bot started successfully!")
        if st.button("Stop COIN Short 2"):
            stop_process('bots', 'COIN_short2')
            st.success("COIN Short 2 bot stopped successfully!")

    with col7:
        st.write("TSLA Long 2")
        if st.button("Start TSLA Long 2"):
            if start_bot('TSLA_long2'):
                st.success("TSLA Long 2 bot started successfully!")
        if st.button("Stop TSLA Long 2"):
            stop_process('bots', 'TSLA_long2')
            st.success("TSLA Long 2 bot stopped successfully!")

    with col8:
        st.write("TSLA Short 2")
        if st.button("Start TSLA Short 2"):
            if start_bot('TSLA_short2'):
                st.success("TSLA Short 2 bot started successfully!")
        if st.button("Stop TSLA Short 2"):
            stop_process('bots', 'TSLA_short2')
            st.success("TSLA Short 2 bot stopped successfully!")

    st.write("-----")
    col_all_start, col_all_stop = st.columns(2)
    with col_all_start:
        if st.button("Start All Bots"):
            success = True
            # Define the bots to start in order
            bots_to_start = [
                'COIN_long',    # bot_id 1
                'COIN_short',   # bot_id 2
                'COIN_long2',   # bot_id 3
                'COIN_short2',  # bot_id 4
                'TSLA_long',    # bot_id 5
                'TSLA_short',   # bot_id 6
                'TSLA_long2',   # bot_id 7
                'TSLA_short2'   # bot_id 8
            ]
            
            st.info("Starting all bots with 5-second intervals to avoid connection issues...")
            
            for bot_name in bots_to_start:
                if not start_bot(bot_name):
                    success = False
                    st.error(f"Failed to start {bot_name}")
                else:
                    st.info(f"Started {bot_name}")
                # Add a longer delay between starting bots to prevent database connection issues
                time.sleep(5)
                
            if success:
                st.success("All bots started successfully!")
            else:
                st.warning("Some bots failed to start. Check the logs for details.")

    with col_all_stop:
        if st.button("Stop All Bots"):
            for bot_name in st.session_state.bot_processes.keys():
                stop_process('bots', bot_name)
            st.success("All bots stopped successfully!")

# Logs Section
with tab_logs:
    st.header("System Logs")

    # Update all logs
    update_logs()
    
    # Create tabs for different log views
    log_tab1, log_tab2, log_tab3, log_tab4, log_tab5 = st.tabs([
        "Bot Logs", "IB Controller Logs", "System Logs", "Historic Logs", "Manual Log Entry"
    ])
    
    with log_tab1:
        st.subheader("Trading Bot Logs")
        
        # Create filters for the logs
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_bot = st.selectbox(
                "Select Bot", 
                options=list(st.session_state.bot_processes.keys()),
                key="log_bot_selector"
            )
        
        with col2:
            log_filter = st.text_input("Filter logs (contains text)", key="log_filter")
            
        with col3:
            max_logs = st.slider("Max logs to display", 10, 500, 100, 10, key="max_logs")
            auto_scroll = st.checkbox("Auto-scroll to bottom", value=True, key="auto_scroll")
        
        # Display logs for the selected bot with filtering
        if selected_bot:
            logs = st.session_state.log_buffer.get(selected_bot, [])
            
            # Apply text filter if provided
            if log_filter:
                logs = [log for log in logs if log_filter.lower() in log.lower()]
            
            # Get the bot status
            bot_status = "Running" if st.session_state.bot_processes.get(selected_bot) else "Stopped"
            status_color = "green" if bot_status == "Running" else "red"
            
            # Display bot status
            st.markdown(f"**Bot Status:** <span style='color:{status_color};'>{bot_status}</span>", unsafe_allow_html=True)
            
            # Create a container for logs with a fixed height and scrolling
            log_container = st.container()
            
            with log_container:
                # Display the logs in a scrollable area
                if logs:
                    # Display only the most recent logs (limited by max_logs)
                    displayed_logs = logs[-max_logs:] if len(logs) > max_logs else logs
                    
                    # Format logs with timestamps if available
                    formatted_logs = []
                    for log in displayed_logs:
                        # Check if log has a timestamp at the beginning
                        if log and (log.startswith("20") or log.startswith("202")):
                            try:
                                # Split by first space to separate timestamp and message
                                parts = log.split(" ", 1)
                                if len(parts) > 1:
                                    timestamp, message = parts
                                    # Format with bold timestamp
                                    formatted_log = f"**{timestamp}** {message}"
                                else:
                                    formatted_log = log
                            except:
                                formatted_log = log
                        else:
                            formatted_log = log
                        
                        # Add color coding based on log level/content
                        if "error" in log.lower() or "exception" in log.lower():
                            formatted_log = f"<span style='color:red;'>{formatted_log}</span>"
                        elif "warning" in log.lower():
                            formatted_log = f"<span style='color:orange;'>{formatted_log}</span>"
                        elif "info" in log.lower():
                            formatted_log = f"<span style='color:blue;'>{formatted_log}</span>"
                        elif "trade" in log.lower() or "position" in log.lower():
                            formatted_log = f"<span style='color:green;'>{formatted_log}</span>"
                            
                        formatted_logs.append(formatted_log)
                    
                    # Join all logs and display with markdown for formatting
                    log_text = "\n\n".join(formatted_logs)
                    st.markdown(log_text, unsafe_allow_html=True)
                    
                    # Add auto-scrolling effect if enabled
                    if auto_scroll and logs:
                        st.markdown(
                            """
                            <script>
                                var element = document.querySelector('[data-testid="stVerticalBlock"]');
                                element.scrollTop = element.scrollHeight;
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info(f"No logs available for {selected_bot}")
            
            # Add buttons for log management
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Clear Logs", key=f"clear_{selected_bot}_logs"):
                    st.session_state.log_buffer[selected_bot] = []
                    st.experimental_rerun()
            
            with col2:
                # Add a button to export logs to a file
                if st.button("Export Logs", key=f"export_{selected_bot}_logs"):
                    log_text = "\n".join(logs)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{selected_bot}_logs_{timestamp}.txt"
                    
                    # Create a download button for the logs
                    st.download_button(
                        label="Download Logs",
                        data=log_text,
                        file_name=filename,
                        mime="text/plain",
                        key=f"download_{selected_bot}_logs"
                    )
            
            with col3:
                # Add a button to save logs to database
                if st.button("Save Logs to DB", key=f"save_{selected_bot}_logs"):
                    success = asyncio.run(save_logs_to_db(selected_bot))
                    if success:
                        st.success(f"Logs for {selected_bot} saved to database")
                    else:
                        st.error(f"Failed to save logs for {selected_bot}")
    
    with log_tab2:
        st.subheader("IB Controller Logs")
        
        # Filter for IB Controller logs
        ib_log_filter = st.text_input("Filter logs (contains text)", key="ib_log_filter")
        ib_max_logs = st.slider("Max logs to display", 10, 500, 100, 10, key="ib_max_logs")
        
        # Get IB Controller logs
        ib_logs = st.session_state.log_buffer['ib_controller']
        
        # Apply filter if provided
        if ib_log_filter:
            ib_logs = [log for log in ib_logs if ib_log_filter.lower() in log.lower()]
        
        # Get status
        ib_status = "Running" if st.session_state.ib_controller_process else "Stopped"
        ib_status_color = "green" if ib_status == "Running" else "red"
        
        # Display IB Controller status
        st.markdown(f"**IB Controller Status:** <span style='color:{ib_status_color};'>{ib_status}</span>", unsafe_allow_html=True)
        
        # Display logs
        if ib_logs:
            # Display only the most recent logs
            displayed_logs = ib_logs[-ib_max_logs:] if len(ib_logs) > ib_max_logs else ib_logs
            
            # Format logs with color coding
            formatted_logs = []
            for log in displayed_logs:
                if "error" in log.lower() or "exception" in log.lower():
                    formatted_log = f"<span style='color:red;'>{log}</span>"
                elif "warning" in log.lower():
                    formatted_log = f"<span style='color:orange;'>{log}</span>"
                else:
                    formatted_log = log
                formatted_logs.append(formatted_log)
            
            log_text = "\n\n".join(formatted_logs)
            st.markdown(log_text, unsafe_allow_html=True)
        else:
            st.info("No IB Controller logs available")
        
        # Add buttons for log management
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear IB Logs"):
                st.session_state.log_buffer['ib_controller'] = []
                st.experimental_rerun()
        
        with col2:
            # Add a button to export logs to a file
            if st.button("Export IB Logs"):
                log_text = "\n".join(ib_logs)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ib_controller_logs_{timestamp}.txt"
                
                # Create a download button for the logs
                st.download_button(
                    label="Download IB Logs",
                    data=log_text,
                    file_name=filename,
                    mime="text/plain"
                )
        
        with col3:
            # Add a button to save logs to database
            if st.button("Save IB Logs to DB"):
                success = asyncio.run(save_logs_to_db('ib_controller'))
                if success:
                    st.success("IB Controller logs saved to database")
                else:
                    st.error("Failed to save IB Controller logs")
    
    with log_tab3:
        st.subheader("System Log Statistics")
        
        # Calculate and display log statistics
        log_stats = {}
        for bot_name, logs in st.session_state.log_buffer.items():
            # Count total logs
            total_logs = len(logs)
            
            # Count error logs
            error_logs = sum(1 for log in logs if "error" in log.lower() or "exception" in log.lower())
            
            # Count warning logs
            warning_logs = sum(1 for log in logs if "warning" in log.lower())
            
            # Count trade-related logs
            trade_logs = sum(1 for log in logs if "trade" in log.lower() or "position" in log.lower())
            
            # Store stats
            log_stats[bot_name] = {
                "total": total_logs,
                "errors": error_logs,
                "warnings": warning_logs,
                "trades": trade_logs
            }
        
        # Convert to DataFrame
        stats_df = pd.DataFrame.from_dict(log_stats, orient='index')
        stats_df = stats_df.reset_index().rename(columns={"index": "Bot/Component"})
        
        # Display statistics
        st.dataframe(stats_df, use_container_width=True)
        
        # Create visualization
        if not stats_df.empty:
            # Create bar chart of errors and warnings
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=stats_df["Bot/Component"],
                y=stats_df["errors"],
                name="Errors",
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=stats_df["Bot/Component"],
                y=stats_df["warnings"],
                name="Warnings",
                marker_color='orange'
            ))
            
            fig.add_trace(go.Bar(
                x=stats_df["Bot/Component"],
                y=stats_df["trades"],
                name="Trade Events",
                marker_color='green'
            ))
            
            fig.update_layout(
                title="Log Event Distribution",
                xaxis_title="Bot/Component",
                yaxis_title="Count",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add buttons for log management
        col1, col2 = st.columns(2)
        with col1:
            # Add options to manage all logs
            if st.button("Clear All Logs"):
                for bot_name in st.session_state.log_buffer:
                    st.session_state.log_buffer[bot_name] = []
                st.experimental_rerun()
        
        with col2:
            # Add a button to save all logs to database
            if st.button("Save All Logs to DB"):
                success = asyncio.run(save_logs_to_db())
                if success:
                    st.success("All logs saved to database")
                else:
                    st.error("Failed to save logs to database")
    
    with log_tab4:
        st.subheader("Historic Logs")
        
        # Create filters for historic logs
        col1, col2 = st.columns(2)
        with col1:
            historic_bot = st.selectbox(
                "Select Bot", 
                options=["All Bots"] + list(st.session_state.bot_processes.keys()),
                key="historic_bot_selector"
            )
        
        with col2:
            historic_limit = st.slider("Max logs to retrieve", 100, 5000, 1000, 100, key="historic_limit")
        
        # Button to load historic logs
        if st.button("Load Historic Logs"):
            with st.spinner("Loading logs from database..."):
                # Handle 'All Bots' selection
                bot_to_load = None if historic_bot == "All Bots" else historic_bot
                # Load logs from database
                historic_logs = asyncio.run(load_logs_from_db(bot_to_load, historic_limit))
                
                if historic_logs:
                    # Create tabs for each bot
                    if len(historic_logs) > 1:
                        historic_bot_tabs = st.tabs(list(historic_logs.keys()))
                        
                        for i, (bot_name, logs) in enumerate(historic_logs.items()):
                            with historic_bot_tabs[i]:
                                st.write(f"**{len(logs)}** log entries for **{bot_name}**")
                                
                                # Filter logs if requested
                                log_filter = st.text_input(
                                    "Filter logs (contains text)", 
                                    key=f"historic_filter_{bot_name}"
                                )
                                
                                if log_filter:
                                    logs = [log for log in logs if log_filter.lower() in log.lower()]
                                
                                # Format logs with timestamps and color coding
                                formatted_logs = []
                                for log in logs:
                                    if "error" in log.lower() or "exception" in log.lower():
                                        formatted_log = f"<span style='color:red;'>{log}</span>"
                                    elif "warning" in log.lower():
                                        formatted_log = f"<span style='color:orange;'>{log}</span>"
                                    elif "info" in log.lower():
                                        formatted_log = f"<span style='color:blue;'>{log}</span>"
                                    elif "trade" in log.lower() or "position" in log.lower():
                                        formatted_log = f"<span style='color:green;'>{log}</span>"
                                    else:
                                        formatted_log = log
                                    formatted_logs.append(formatted_log)
                                
                                log_text = "\n\n".join(formatted_logs)
                                st.markdown(log_text, unsafe_allow_html=True)
                                
                                # Export button
                                if st.button(f"Export {bot_name} Logs", key=f"export_historic_{bot_name}"):
                                    log_text_plain = "\n".join(logs)
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"{bot_name}_historic_logs_{timestamp}.txt"
                                    
                                    st.download_button(
                                        label=f"Download {bot_name} Historic Logs",
                                        data=log_text_plain,
                                        file_name=filename,
                                        mime="text/plain",
                                        key=f"download_historic_{bot_name}"
                                    )
                    else:
                        # Single bot view
                        bot_name = list(historic_logs.keys())[0]
                        logs = historic_logs[bot_name]
                        
                        st.write(f"**{len(logs)}** log entries for **{bot_name}**")
                        
                        # Filter logs if requested
                        log_filter = st.text_input("Filter logs (contains text)", key="historic_filter_single")
                        
                        if log_filter:
                            logs = [log for log in logs if log_filter.lower() in log.lower()]
                        
                        # Format logs with timestamps and color coding
                        formatted_logs = []
                        for log in logs:
                            if "error" in log.lower() or "exception" in log.lower():
                                formatted_log = f"<span style='color:red;'>{log}</span>"
                            elif "warning" in log.lower():
                                formatted_log = f"<span style='color:orange;'>{log}</span>"
                            elif "info" in log.lower():
                                formatted_log = f"<span style='color:blue;'>{log}</span>"
                            elif "trade" in log.lower() or "position" in log.lower():
                                formatted_log = f"<span style='color:green;'>{log}</span>"
                            else:
                                formatted_log = log
                            formatted_logs.append(formatted_log)
                        
                        log_text = "\n\n".join(formatted_logs)
                        st.markdown(log_text, unsafe_allow_html=True)
                        
                        # Export button
                        if st.button(f"Export {bot_name} Logs", key="export_historic_single"):
                            log_text_plain = "\n".join(logs)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{bot_name}_historic_logs_{timestamp}.txt"
                            
                            st.download_button(
                                label=f"Download {bot_name} Historic Logs",
                                data=log_text_plain,
                                file_name=filename,
                                mime="text/plain",
                                key="download_historic_single"
                            )
                else:
                    st.info("No historic logs found in the database")
        
        # Add a button to clear historic logs from database
        if st.button("Clear Historic Logs from Database"):
            if st.session_state.get('confirm_clear_historic', False):
                # User already confirmed, proceed with deletion
                try:
                    async def clear_historic_logs():
                        async with asyncpg.create_pool(**DB_CONFIG) as pool:
                            if historic_bot == "All Bots":
                                await pool.execute("TRUNCATE TABLE bot_logs;")
                                message = "All historic logs cleared from database"
                            else:
                                await pool.execute(
                                    "DELETE FROM bot_logs WHERE bot_name = $1;",
                                    historic_bot
                                )
                                message = f"Historic logs for {historic_bot} cleared from database"
                            return message
                    
                    message = asyncio.run(clear_historic_logs())
                    st.success(message)
                    # Reset confirmation state
                    st.session_state.confirm_clear_historic = False
                except Exception as e:
                    st.error(f"Error clearing historic logs: {e}")
            else:
                # Ask for confirmation
                st.warning(f"Are you sure you want to clear {'all' if historic_bot == 'All Bots' else historic_bot} historic logs? This cannot be undone!")
                st.session_state.confirm_clear_historic = True
                if st.button("Yes, Clear Historic Logs"):
                    # User confirmed, will be handled on next rerun
                    st.experimental_rerun()
                if st.button("No, Cancel"):
                    st.session_state.confirm_clear_historic = False
                    st.experimental_rerun()

    # Add new tab for manual log entry
    with log_tab5:
        st.subheader("Manual Log Entry")
        
        st.write("""
        Use this section to manually add log entries for your bots. 
        This is useful for adding notes, comments, or annotations about bot behavior or trading decisions.
        """)
        
        # Create form for log entry
        with st.form(key="manual_log_form"):
            # Bot selection
            target_bot = st.selectbox(
                "Select Target Bot",
                options=list(st.session_state.bot_processes.keys()),
                key="manual_log_bot"
            )
            
            # Log level selection
            log_level = st.selectbox(
                "Log Level",
                options=["INFO", "WARNING", "ERROR", "TRADE", "NOTE"],
                index=0,
                key="manual_log_level"
            )
            
            # Log message input
            log_message = st.text_area(
                "Log Message",
                height=100,
                key="manual_log_message"
            )
            
            # Submit button
            submit_button = st.form_submit_button(label="Add Log Entry")
        
        # Process form submission
        if submit_button and target_bot and log_message:
            # Add the log entry
            log_bot_event(target_bot, log_level, log_message)
            st.success(f"Log entry added for {target_bot}")
            
            # Provide option to save to database
            if st.button("Save New Log to Database"):
                success = asyncio.run(save_logs_to_db(target_bot))
                if success:
                    st.success(f"Logs for {target_bot} saved to database")
                else:
                    st.error(f"Failed to save logs for {target_bot}")
        
        # Add a separator
        st.markdown("---")
        
        # Add section for batch log entry from file
        st.subheader("Import Logs from File")
        
        uploaded_file = st.file_uploader("Choose a log file", type=['txt', 'log'])
        
        if uploaded_file is not None:
            # Import target selection
            import_target = st.selectbox(
                "Import logs to",
                options=list(st.session_state.bot_processes.keys()),
                key="import_log_target"
            )
            
            # Parse log file
            if st.button("Import Logs"):
                try:
                    # Read and decode the file
                    log_content = uploaded_file.read().decode('utf-8')
                    log_lines = log_content.splitlines()
                    
                    # Count lines
                    line_count = len(log_lines)
                    
                    # Ask for confirmation if file is large
                    if line_count > 1000:
                        st.warning(f"This file contains {line_count} lines. Importing large files may affect performance.")
                        confirm_import = st.checkbox("Import anyway")
                        if not confirm_import:
                            st.stop()
                    
                    # Process each line
                    for line in log_lines:
                        if line.strip():
                            # Add to log buffer
                            if import_target not in st.session_state.log_buffer:
                                st.session_state.log_buffer[import_target] = []
                            
                            # Add timestamp if missing
                            if not (line.startswith("20") or line.startswith("202")):
                                line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [IMPORTED] {line}"
                            
                            st.session_state.log_buffer[import_target].append(line)
                    
                    st.success(f"Successfully imported {line_count} log lines for {import_target}")
                except Exception as e:
                    st.error(f"Error importing logs: {e}")
        
        # Template section
        st.markdown("---")
        st.subheader("Log Templates")
        
        # Define some common log templates
        templates = {
            "Trade Entry": "{timestamp} [TRADE] Entered {long_short} position for {ticker} at {price}",
            "Trade Exit": "{timestamp} [TRADE] Exited {long_short} position for {ticker} at {price}, PnL: {pnl}",
            "Strategy Change": "{timestamp} [INFO] Changed strategy parameters for {ticker}: {params}",
            "Position Adjustment": "{timestamp} [INFO] Adjusted position size for {ticker} from {old_size} to {new_size}",
            "Error Note": "{timestamp} [ERROR] Bot encountered an issue: {error_details}"
        }
        
        # Template selection
        selected_template = st.selectbox(
            "Select Template",
            options=list(templates.keys()),
            key="log_template"
        )
        
        # Show the template
        st.code(templates[selected_template], language="text")
        
        # Button to use the template
        if st.button("Use This Template"):
            # Pre-fill the log message with the template
            template_text = templates[selected_template].replace("{timestamp}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            st.session_state.manual_log_message = template_text
            st.experimental_rerun()

# Tables Section
with tab_tables:
    st.header("Database Tables")

    async def fetch_data():
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Fetch trade data
            trades = await pool.fetch("""
                SELECT * FROM sim_bot_trades 
                ORDER BY entry_time DESC 
                LIMIT 100
            """)
            return trades

    if st.button("Refresh Data"):
        try:
            trades = asyncio.run(fetch_data())
            
            # Display trade data
            st.subheader("Recent Trades")
            if trades and len(trades) > 0:
                # Convert the asyncpg records to dictionaries before building DataFrame
                trades_df = pd.DataFrame([dict(t) for t in trades])
                
                # Format timestamps for better display
                if 'entry_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                if 'exit_time' in trades_df.columns:
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                
                # Set trade_id as index if it exists
                if 'trade_id' in trades_df.columns:
                    trades_df.set_index('trade_id', inplace=True)
                
                st.dataframe(trades_df)
            else:
                st.info("No trades found in the database.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Trade Data Section
with tab_trades:
    st.header("Trade Analysis")

    # Add tabs for different analysis sections
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "Trade Statistics", "Bot Metrics", "Variable Weights"
    ])

    with analysis_tab1:
        async def fetch_trade_stats():
            try:
                async with asyncpg.create_pool(**DB_CONFIG) as pool:
                    # Debug: Check if table exists
                    table_exists = await pool.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'sim_bot_trades'
                        );
                    """)
                    st.write(f"sim_bot_trades table exists: {table_exists}")
                    
                    if not table_exists:
                        st.error("sim_bot_trades table does not exist!")
                        return None, None, None

                    # Debug: Show table structure
                    columns = await pool.fetch("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'sim_bot_trades';
                    """)
                    st.write("Table structure:", [col['column_name'] for col in columns])

                    # First check if there are any trades
                    trades_count = await pool.fetchval("""
                        SELECT COUNT(*) 
                        FROM sim_bot_trades;
                    """)
                    st.write(f"Total trades in database: {trades_count}")
                    
                    closed_trades_count = await pool.fetchval("""
                        SELECT COUNT(*) 
                        FROM sim_bot_trades 
                        WHERE trade_status = 'closed';
                    """)
                    st.write(f"Closed trades in database: {closed_trades_count}")
                    
                    if closed_trades_count == 0:
                        st.warning("No closed trades found in the database.")
                        return None, None, None

                    # Get trade statistics with debug info
                    stats = await pool.fetch("""
                        WITH trade_stats AS (
                            SELECT 
                                bot_id,
                                ticker,
                                COUNT(*) as trade_count,
                                COUNT(CASE WHEN trade_pnl > 0 THEN 1 END) as winning_trades,
                                COUNT(CASE WHEN trade_pnl <= 0 THEN 1 END) as losing_trades,
                                ROUND(AVG(CASE WHEN trade_pnl IS NOT NULL THEN trade_pnl ELSE 0 END)::numeric, 2) as avg_pnl,
                                ROUND(SUM(CASE WHEN trade_pnl IS NOT NULL THEN trade_pnl ELSE 0 END)::numeric, 2) as total_pnl,
                                ROUND(AVG(CASE 
                                    WHEN exit_time IS NOT NULL AND entry_time IS NOT NULL 
                                    THEN EXTRACT(EPOCH FROM (exit_time - entry_time)) 
                                    ELSE 0 
                                END)::numeric, 2) as avg_duration,
                                ROUND(
                                    (COUNT(CASE WHEN trade_pnl > 0 THEN 1 END)::float / 
                                    NULLIF(COUNT(*), 0)::float * 100)::numeric, 
                                    2
                                ) as calculated_win_rate
                            FROM sim_bot_trades
                            WHERE trade_status = 'closed'
                            GROUP BY bot_id, ticker
                        )
                        SELECT * FROM trade_stats
                        ORDER BY total_pnl DESC;
                    """)
                    
                    if not stats:
                        st.warning("Query returned no results.")
                        return None, None, None

                    # Debug: Show what columns we got back
                    if len(stats) > 0:
                        st.write("Columns in result:", stats[0].keys())
                        st.write(f"Number of bot/ticker combinations found: {len(stats)}")
                    
                    return stats, None, None
                    
            except Exception as e:
                st.error(f"Database error: {str(e)}")
                st.error("Full error details:")
                st.exception(e)
                return None, None, None

        if st.button("Calculate Statistics", key="calc_stats_button"):
            try:
                stats, _, _ = asyncio.run(fetch_trade_stats())
                
                if stats and len(stats) > 0:
                    # Debug: Show raw stats data
                    st.write("Raw statistics data:", stats)
                    
                    # Create DataFrame with explicit column names
                    stats_df = pd.DataFrame([dict(row) for row in stats])
                    
                    # Debug: Show DataFrame info
                    st.write("DataFrame columns:", stats_df.columns.tolist())
                    st.write("DataFrame shape:", stats_df.shape)
                    
                    # Display summary statistics
                    st.subheader("Trading Summary")
                    total_trades = stats_df['trade_count'].sum()
                    total_pnl = stats_df['total_pnl'].sum()
                    avg_win_rate = stats_df['calculated_win_rate'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trades", f"{total_trades:,}")
                    with col2:
                        st.metric("Total PnL", f"${total_pnl:,.2f}")
                    with col3:
                        st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
                    
                    # Display detailed statistics by bot
                    st.subheader("Detailed Statistics by Bot")
                    st.dataframe(
                        stats_df.style.format({
                            'trade_count': lambda x: safe_format(x, '{:,}'),
                            'winning_trades': lambda x: safe_format(x, '{:,}'),
                            'losing_trades': lambda x: safe_format(x, '{:,}'),
                            'avg_pnl': lambda x: safe_format(x, '${:.2f}'),
                            'total_pnl': lambda x: safe_format(x, '${:.2f}'),
                            'calculated_win_rate': lambda x: safe_format(x, '{:.1f}%'),
                            'avg_duration': lambda x: safe_format(x, '{:.1f}')
                        }),
                        use_container_width=True
                    )
                    
                    # Create visualizations
                    st.subheader("Performance Visualization")
                    
                    # PnL Distribution
                    fig = go.Figure()
                    for bot_id in stats_df['bot_id'].unique():
                        bot_data = stats_df[stats_df['bot_id'] == bot_id]
                        fig.add_trace(go.Bar(
                            name=f'Bot {bot_id}',
                            x=bot_data['ticker'],
                            y=bot_data['total_pnl'],
                            text=bot_data['total_pnl'].apply(lambda x: f'${x:,.2f}'),
                            textposition='auto',
                        ))
                    
                    fig.update_layout(
                        title='Total PnL by Bot and Ticker',
                        xaxis_title='Ticker',
                        yaxis_title='Total PnL ($)',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Win Rate vs PnL Scatter
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stats_df['calculated_win_rate'],
                        y=stats_df['avg_pnl'],
                        mode='markers+text',
                        text=stats_df.apply(lambda x: f"Bot {x['bot_id']} - {x['ticker']}", axis=1),
                        textposition="top center",
                        marker=dict(
                            size=stats_df['trade_count'],
                            sizeref=2.*max(stats_df['trade_count'])/(40.**2),
                            sizemin=4
                        )
                    ))
                    
                    fig.update_layout(
                        title='Win Rate vs Average PnL',
                        xaxis_title='Win Rate (%)',
                        yaxis_title='Average PnL ($)',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No trade statistics available. Please check if there are closed trades in the database.")
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
                st.error("Please check the database connection and make sure the sim_bot_trades table exists with trade data.")

    with analysis_tab2:
        st.subheader("Bot Metrics Management")
        
        # Add tabs for different metric views
        metric_tab1, metric_tab2, metric_tab3 = st.tabs([
            "Performance Overview", "Advanced Metrics", "Real-time Monitor"
        ])
        
        async def fetch_bot_metrics():
            try:
                async with asyncpg.create_pool(**DB_CONFIG) as pool:
                    # First check if the table exists
                    table_exists = await pool.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'bot_metrics'
                        );
                    """)
                    
                    if not table_exists:
                        st.warning("Bot metrics table does not exist. Please run the metrics calculator first.")
                        return None
                    
                    # Get the actual columns from the table
                    columns = await pool.fetch("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'bot_metrics';
                    """)
                    column_names = [col['column_name'] for col in columns]
                    
                    # Build the query dynamically based on available columns
                    base_columns = ['bot_id', 'ticker']
                    metric_columns = [
                        'one_hour_performance', 'two_hour_performance',
                        'one_day_performance', 'one_week_performance',
                        'one_month_performance', 'avg_win_rate',
                        'avg_drawdown', 'max_drawdown', 'profit_factor',
                        'avg_profit_per_trade', 'total_pnl'
                    ]
                    
                    # Only include columns that exist in the table
                    select_columns = base_columns + [col for col in metric_columns if col in column_names]
                    
                    # Construct the query
                    query = f"""
                        SELECT {', '.join(select_columns)}
                        FROM bot_metrics
                        ORDER BY bot_id, ticker;
                    """
                    
                    metrics = await pool.fetch(query)
                    return [dict(m) for m in metrics]
            except Exception as e:
                st.error(f"Error fetching bot metrics: {str(e)}")
                return None

        with metric_tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("Refresh Performance Metrics"):
                    metrics = asyncio.run(fetch_bot_metrics())
                    if metrics:
                        metrics_df = pd.DataFrame(metrics)
                        
                        # Format numeric columns that we know exist
                        numeric_cols = [col for col in metrics_df.columns if any(
                            metric in col for metric in ['performance', 'rate', 'drawdown', 'factor', 'pnl']
                        )]
                        
                        for col in numeric_cols:
                            if col in metrics_df.columns:
                                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
                                if any(metric in col for metric in ['performance', 'rate', 'drawdown']):
                                    # Normalize percentage values - divide by 100 for large values and cap at +/- 100%
                                    # This fixes the unusually large percentage values
                                    metrics_df[col] = metrics_df[col].apply(
                                        lambda x: max(min(x, 100), -100) if abs(x) <= 100 
                                        else max(min(x/100, 100), -100)
                                    )
                                    metrics_df[col] = metrics_df[col].map('{:.2%}'.format)
                                else:
                                    metrics_df[col] = metrics_df[col].map('{:.2f}'.format)
                        
                        st.dataframe(metrics_df)
                        
                        # Create performance heatmap for available performance metrics
                        performance_cols = [col for col in metrics_df.columns if 'performance' in col]
                        if performance_cols and 'bot_id' in metrics_df.columns and 'ticker' in metrics_df.columns:
                            try:
                                heatmap_data = metrics_df[['bot_id', 'ticker'] + performance_cols].copy()
                                
                                # Check for empty dataframe
                                if not heatmap_data.empty:
                                    for col in performance_cols:
                                        # Convert percentage strings to numeric values safely
                                        heatmap_data[col] = safe_pct_to_numeric(heatmap_data[col])
                                        # Normalize percentage values for visualization
                                        heatmap_data[col] = heatmap_data[col].apply(
                                            lambda x: max(min(x, 100), -100) if abs(x) <= 100 
                                            else max(min(x/100, 100), -100)
                                        )
                                    
                                    # Create heatmap
                                    fig = go.Figure(data=go.Heatmap(
                                        z=heatmap_data[performance_cols].values,
                                        x=performance_cols,
                                        y=heatmap_data.apply(lambda x: f"Bot {x['bot_id']} - {x['ticker']}", axis=1),
                                        colorscale='RdYlGn'
                                    ))
                                    fig.update_layout(title='Performance Heatmap Across Timeframes')
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No data available for performance heatmap visualization.")
                            except Exception as e:
                                st.warning(f"Error creating performance heatmap: {str(e)}")
                        else:
                            st.info("Performance heatmap data is not available. Required performance columns are missing.")
            
            with col2:
                st.write("Quick Stats")
                if 'metrics_df' in locals():
                    try:
                        # Calculate and display key statistics for available metrics
                        if 'one_day_performance' in metrics_df.columns and not metrics_df['one_day_performance'].empty:
                            # Convert to numeric safely and find the max
                            perf_values = safe_pct_to_numeric(metrics_df['one_day_performance'])
                            if not perf_values.empty and not perf_values.isna().all():
                                # Normalize percentage values
                                perf_values = perf_values.apply(
                                    lambda x: max(min(x, 100), -100) if abs(x) <= 100 
                                    else max(min(x/100, 100), -100)
                                )
                                idx = perf_values.idxmax()
                                best_performer = metrics_df.loc[idx]
                                # Format the value as percentage
                                perf_display = f"{perf_values[idx]:.2%}"
                                st.metric("Best 24h Performer", 
                                        f"Bot {best_performer['bot_id']} - {best_performer['ticker']}", 
                                        perf_display)
                        
                        if 'avg_win_rate' in metrics_df.columns and not metrics_df['avg_win_rate'].empty:
                            # Convert to numeric safely and find the max
                            win_values = safe_pct_to_numeric(metrics_df['avg_win_rate'])
                            if not win_values.empty and not win_values.isna().all():
                                # Normalize percentage values
                                win_values = win_values.apply(
                                    lambda x: max(min(x, 100), 0) if x <= 100 else x/100
                                )
                                idx = win_values.idxmax()
                                highest_win_rate = metrics_df.loc[idx]
                                # Format the value as percentage
                                win_rate_display = f"{win_values[idx]:.2%}"
                                st.metric("Highest Win Rate", 
                                        f"Bot {highest_win_rate['bot_id']} - {highest_win_rate['ticker']}", 
                                        win_rate_display)
                        
                        if 'total_pnl' in metrics_df.columns and not metrics_df['total_pnl'].empty:
                            total_pnl = pd.to_numeric(metrics_df['total_pnl'], errors='coerce').sum()
                            st.metric("Total System PnL", 
                                    f"${total_pnl:.2f}")
                    except Exception as e:
                        st.warning(f"Could not calculate some statistics: {str(e)}")

        with metric_tab2:
            st.write("Advanced Performance Analytics")
            
            if st.button("Calculate Advanced Metrics"):
                metrics = asyncio.run(fetch_bot_metrics())
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                    
                    # Create Sharpe Ratio vs Drawdown scatter plot
                    fig = go.Figure()
                    
                    # Check if required columns exist
                    if all(col in metrics_df.columns for col in ['profit_factor', 'max_drawdown', 'ticker', 'bot_id']):
                        for ticker in metrics_df['ticker'].unique():
                            ticker_data = metrics_df[metrics_df['ticker'] == ticker]
                            
                            # Convert string percentages to float
                            profit_factor = safe_pct_to_numeric(ticker_data['profit_factor'])
                            drawdown = safe_pct_to_numeric(ticker_data['max_drawdown'])
                            
                            fig.add_trace(go.Scatter(
                                x=drawdown,
                                y=profit_factor,
                                mode='markers+text',
                                name=ticker,
                                text=ticker_data['bot_id'],
                                textposition="top center"
                            ))
                        
                        fig.update_layout(
                            title='Risk-Reward Analysis',
                            xaxis_title='Maximum Drawdown (%)',
                            yaxis_title='Profit Factor',
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Risk-reward analysis data is not available. Some required columns (profit_factor, max_drawdown) are missing from the metrics.")
                    
                    # Win Streak Analysis
                    streak_cols = ['win_streak_2', 'win_streak_3', 'win_streak_4', 'win_streak_5']
                    
                    # Check if at least some of the streak columns exist
                    available_streak_cols = [col for col in streak_cols if col in metrics_df.columns]
                    
                    if available_streak_cols and 'bot_id' in metrics_df.columns and 'ticker' in metrics_df.columns:
                        streak_data = metrics_df[['bot_id', 'ticker'] + available_streak_cols].copy()
                        
                        for col in available_streak_cols:
                            # Convert percentage strings to numeric values safely
                            streak_data[col] = safe_pct_to_numeric(streak_data[col])
                        
                        fig = go.Figure()
                        for idx, row in streak_data.iterrows():
                            fig.add_trace(go.Bar(
                                name=f"Bot {row['bot_id']} - {row['ticker']}",
                                x=[col.replace('win_streak_', '') + ' Wins' for col in available_streak_cols],
                                y=[row[col] for col in available_streak_cols]
                            ))
                        
                        fig.update_layout(
                            title='Win Streak Probability Analysis',
                            barmode='group',
                            xaxis_title='Streak Length',
                            yaxis_title='Probability (%)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Win streak data is not available. Some required columns are missing from the metrics.")

        with metric_tab3:
            st.write("Real-time Performance Monitor")
            
            # Add auto-refresh functionality
            auto_refresh = st.checkbox("Enable Auto-refresh (10s)")
            
            if auto_refresh:
                st.write("Auto-refreshing every 10 seconds...")
                time.sleep(10)  # Simple implementation - in production use async
            
            if st.button("Refresh Monitor") or auto_refresh:
                metrics = asyncio.run(fetch_bot_metrics())
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                    
                    # Create real-time performance gauge charts
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Hour Performance Gauge
                        if 'one_hour_performance' in metrics_df.columns:
                            # Normalize the performance value
                            hour_perf_raw = safe_pct_to_numeric(metrics_df['one_hour_performance']).mean()
                            hour_perf = max(min(hour_perf_raw, 100), -100) if abs(hour_perf_raw) <= 100 else max(min(hour_perf_raw/100, 100), -100)
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = hour_perf,
                                title = {'text': "1h Performance"},
                                gauge = {'axis': {'range': [-5, 5]},
                                        'bar': {'color': "darkblue"},
                                        'steps' : [
                                            {'range': [-5, 0], 'color': "lightgray"},
                                            {'range': [0, 5], 'color': "gray"}]}))
                            st.plotly_chart(fig)
                        else:
                            st.info("1h performance data is not available.")
                    
                    with col2:
                        # Win Rate Gauge
                        if 'avg_win_rate' in metrics_df.columns:
                            # Normalize the win rate value
                            win_rate_raw = safe_pct_to_numeric(metrics_df['avg_win_rate']).mean()
                            win_rate = max(min(win_rate_raw, 100), 0) if win_rate_raw <= 100 else win_rate_raw/100
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = win_rate,
                                title = {'text': "Win Rate"},
                                gauge = {'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkgreen"}}))
                            st.plotly_chart(fig)
                        else:
                            st.info("Win rate data is not available.")
                    
                    with col3:
                        # Profit Factor Gauge
                        if 'profit_factor' in metrics_df.columns:
                            # Normalize profit factor to a reasonable range (0-10)
                            profit_factor_raw = safe_pct_to_numeric(metrics_df['profit_factor']).mean()
                            profit_factor = min(profit_factor_raw, 10) if profit_factor_raw <= 10 else profit_factor_raw/10
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = profit_factor,
                                title = {'text': "Profit Factor"},
                                gauge = {'axis': {'range': [0, 3]},
                                        'bar': {'color': "darkorange"}}))
                            st.plotly_chart(fig)
                        else:
                            st.info("Profit factor data is not available.")
                    
                    # Add real-time trade frequency chart
                    if 'trade_frequency' in metrics_df.columns:
                        trade_freq = pd.to_numeric(metrics_df['trade_frequency'], errors='coerce')
                        fig = go.Figure(go.Bar(
                            x=metrics_df.apply(lambda x: f"Bot {x['bot_id']} - {x['ticker']}", axis=1),
                            y=trade_freq,
                            marker_color='lightblue'
                        ))
                        fig.update_layout(
                            title='Current Trade Frequency by Bot',
                            xaxis_title='Bot',
                            yaxis_title='Trades per Hour'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Trade frequency data is not available. The 'trade_frequency' column is missing from the metrics.")

    with analysis_tab3:
        st.subheader("Variable Weights Management")
        
        # Create two columns for the layout
        weight_col1, weight_col2 = st.columns([2, 1])
        
        with weight_col1:
            # Current Weights Visualization
            st.write("Current Variable Weights Distribution")
            
            # Define the default weights for all variables
            default_weights = {
                'avg_drawdown': 8.0,
                'avg_win_rate': 10.0,
                'one_day_performance': 10.0,
                'one_hour_performance': 15.0,
                'one_month_performance': 5.0,
                'one_week_performance': 7.5,
                'price_model_score': 5.0,
                'price_wall_score': 3.0,
                'profit_per_second': 12.0,
                'two_hour_performance': 12.5,
                'volume_model_score': 5.0,
                'win_streak_2': 2.0,
                'win_streak_3': 1.5,
                'win_streak_4': 1.5,
                'win_streak_5': 1.0
            }

            async def fetch_variable_weights():
                try:
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        # Check if weights table exists
                        table_exists = await pool.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'variable_weights'
                            );
                        """)
                        
                        if not table_exists:
                            # Create table if it doesn't exist
                            await pool.execute("""
                                CREATE TABLE IF NOT EXISTS variable_weights (
                                    weight_id SERIAL PRIMARY KEY,
                                    variable_name VARCHAR(50) NOT NULL UNIQUE,
                                    weight DECIMAL(4,1) NOT NULL,
                                    last_updated TIMESTAMP DEFAULT NOW()
                                );
                            """)
                            
                            # Insert default weights
                            for var_name, weight in default_weights.items():
                                await pool.execute("""
                                    INSERT INTO variable_weights (variable_name, weight)
                                    VALUES ($1, $2)
                                    ON CONFLICT (variable_name) DO NOTHING;
                                """, var_name, weight)
                        
                        # Fetch current weights
                        weights = await pool.fetch("""
                            SELECT variable_name, weight, last_updated
                            FROM variable_weights
                            ORDER BY weight DESC;
                        """)
                        return weights
                except Exception as e:
                    st.error(f"Error fetching variable weights: {str(e)}")
                    return None

            if st.button("Refresh Variable Weights"):
                weights = asyncio.run(fetch_variable_weights())
                if weights:
                    # Convert to DataFrame and ensure column names are correct
                    weights_df = pd.DataFrame(weights, columns=['variable_name', 'weight', 'last_updated'])
                    
                    # Create bar chart of weights
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=weights_df['weight'],
                        y=weights_df['variable_name'],
                        orientation='h',
                        text=weights_df['weight'].apply(lambda x: f'{x/100:.2f}'),
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title='Variable Weights Distribution',
                        xaxis_title='Weight (Decimal)',
                        yaxis_title='Variable Name',
                        height=600,
                        showlegend=False,
                        xaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display weights table with decimal format
                    weights_df['weight_decimal'] = weights_df['weight'] / 100
                    st.dataframe(
                        weights_df[['variable_name', 'weight_decimal', 'last_updated']].style.format({
                            'weight_decimal': lambda x: safe_format(x, '{:.2f}'),
                            'last_updated': lambda x: safe_format(x, '{:%Y-%m-%d %H:%M:%S}')
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No variable weights data available")

        with weight_col2:
            # Weight Update Form
            st.write("Update Variable Weight")
            
            # Variable selection
            variable_name = st.selectbox(
                "Select Variable",
                list(default_weights.keys())
            )
            
            # Weight input
            weight = st.number_input(
                "Weight (Decimal)",
                min_value=0.0,
                max_value=1.0,
                value=default_weights.get(variable_name, 5.0)/100,
                step=0.01,
                help="Enter the weight as a decimal (0-1)"
            )
            
            # Update button
            if st.button("Update Weight"):
                try:
                    async def update_variable_weight():
                        async with asyncpg.create_pool(**DB_CONFIG) as pool:
                            # First check if the variable exists
                            var_exists = await pool.fetchval("""
                                SELECT EXISTS (
                                    SELECT 1 FROM variable_weights WHERE variable_name = $1
                                )
                            """, variable_name)
                            
                            if var_exists:
                                # Update existing record
                                await pool.execute("""
                                    UPDATE variable_weights
                                    SET weight = $2, last_updated = NOW()
                                    WHERE variable_name = $1
                                """, variable_name, weight)
                            else:
                                # Insert new record
                                await pool.execute("""
                                    INSERT INTO variable_weights (variable_name, weight, last_updated)
                                    VALUES ($1, $2, NOW())
                                """, variable_name, weight)
                            
                            return True
                    
                    asyncio.run(update_variable_weight())
                    st.success(f"Weight for {variable_name} updated to {weight:.2f}")
                except Exception as e:
                    st.error(f"Error updating weight: {str(e)}")
            
            # Show total weight info
            st.write("---")
            st.write("Weight Distribution Info")
            
            async def get_total_weight():
                try:
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        total = await pool.fetchval("""
                            SELECT SUM(weight) FROM variable_weights;
                        """)
                        return (total or 0) / 100  # Convert to decimal
                except Exception as e:
                    st.error(f"Error calculating total weight: {str(e)}")
                    return 0
            
            total_weight = asyncio.run(get_total_weight())
            st.metric(
                "Total Weight",
                f"{total_weight:.2f}",
                delta=f"{total_weight - 1:.2f}" if abs(total_weight - 1) > 0.001 else None,
                delta_color="inverse"
            )
            
            if abs(total_weight - 1) > 0.001:
                st.warning("⚠️ Total weight should sum to 1.0")
            else:
                st.success("✅ Weights are properly distributed")

# Parameters Section
with tab_params:
    st.header("System Parameters")
    col1, col2 = st.columns(2)

    with col1:
        risk_per_trade = st.number_input("Risk Per Trade (%)", 0.0, 5.0, 1.0, 0.1)
        stop_loss = st.number_input("Stop Loss (%)", 0.0, 10.0, 2.0, 0.1)

    with col2:
        take_profit = st.number_input("Take Profit (%)", 0.0, 20.0, 4.0, 0.1)
        max_positions = st.number_input("Max Open Positions", 1, 10, 3)

    if st.button("Save Parameters"):
        # TODO: Implement parameter saving logic
        st.success("Parameters saved successfully!")

# Data Export Section
with tab_export:
    st.header("Data Export")
    st.write("Use the buttons below to export trade or tick data to CSV.")

    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("Export All Trades to CSV"):
            try:
                # Run the export_all_trades.py script
                subprocess.Popen(['python', 'export_all_trades.py'])
                st.success("Exporting all trades... Check the console or logs for status.")
            except Exception as e:
                st.error(f"Error running export_all_trades.py: {e}")

        # Add a download button for the all_trades.csv file
        try:
            with open('all_trades.csv', 'rb') as file:
                st.download_button(
                    label="Download All Trades CSV",
                    data=file,
                    file_name='all_trades.csv',
                    mime='text/csv'
                )
        except FileNotFoundError:
            st.warning("The all_trades.csv file is not available for download.")

    with col_export2:
        if st.button("Export Tick Data to CSV"):
            try:
                # Run the export_tick_data.py script
                subprocess.Popen(['python', 'export_tick_data.py'])
                st.success("Exporting tick data... Check the console or logs for status.")
            except Exception as e:
                st.error(f"Error running export_tick_data.py: {e}")

        # Add a download button for the tick_data.csv file
        try:
            with open('tick_data.csv', 'rb') as file:
                st.download_button(
                    label="Download Tick Data CSV",
                    data=file,
                    file_name='tick_data.csv',
                    mime='text/csv'
                )
        except FileNotFoundError:
            st.warning("The tick_data.csv file is not available for download.")

# Bot Rankings Section
with tab_rankings:
    st.header("Bot Rankings and Fund Allocation")
    
    # Create sub-tabs for different ranking views
    rank_tab1, rank_tab2, rank_tab3, rank_tab4, rank_tab5 = st.tabs([
        "Current Rankings", "Historical Performance", "Weight Management", "Database Diagnostics", "Automated Fund Management"
    ])
    
    # Function to fetch bot rankings
    async def fetch_bot_rankings():
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # Check if table exists
                table_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'bot_rankings'
                    );
                """)
                
                if not table_exists:
                    return None
                    
                # Get current rankings
                rankings = await pool.fetch("""
                    SELECT br.*, bm.ticker, bm.one_day_performance, bm.avg_win_rate, bm.profit_factor
                    FROM bot_rankings br
                    LEFT JOIN bot_metrics bm ON br.bot_id = bm.bot_id
                    WHERE bm.timestamp = (
                        SELECT MAX(timestamp) FROM bot_metrics WHERE bot_id = br.bot_id
                    )
                    ORDER BY br.rank_score DESC;
                """)
                return rankings
        except Exception as e:
            st.error(f"Error fetching bot rankings: {str(e)}")
            return None
    
    # Function to fetch historical ranking data
    async def fetch_historical_rankings(days=30):
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # Get historical rankings with daily resolution
                # Use string concatenation for the interval instead of a parameter
                query = f"""
                    WITH daily_rankings AS (
                        SELECT 
                            bot_id, 
                            DATE(timestamp) as date,
                            rank_score,
                            ROW_NUMBER() OVER (PARTITION BY bot_id, DATE(timestamp) ORDER BY timestamp DESC) as rn
                        FROM bot_rankings
                        WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
                    )
                    SELECT 
                        dr.bot_id, 
                        dr.date, 
                        dr.rank_score,
                        bm.ticker
                    FROM daily_rankings dr
                    LEFT JOIN bot_metrics bm ON dr.bot_id = bm.bot_id
                    WHERE dr.rn = 1
                    ORDER BY dr.date, dr.rank_score DESC
                """
                rankings = await pool.fetch(query)
                return rankings
        except Exception as e:
            st.error(f"Error fetching historical rankings: {str(e)}")
            return None
    
    # Function to fetch fund allocation
    async def fetch_fund_allocation(total_funds=10000):
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # First check if we can import the BotRanker
                try:
                    # Add the parent directory to the path to find the src module
                    import sys
                    import os
                    # Get the current directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # Go up two levels (from user_interface/src to the project root)
                    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
                    # Add to path if not already there
                    if project_root not in sys.path:
                        sys.path.insert(0, project_root)
                    
                    # Now import the BotRanker
                    from src.bot_ranker import BotRanker
                    
                    # Create a bot ranker instance
                    ranker = BotRanker(pool)
                    
                    # Get fund allocation
                    allocations = await ranker.get_fund_allocation(total_funds)
                    return allocations
                except ImportError as e:
                    st.error(f"Error importing BotRanker: {str(e)}")
                    
                    # Fallback: calculate a simple allocation based on rank position
                    
                    # First check if is_active column exists
                    column_exists = await pool.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_name = 'bot_rankings' AND column_name = 'is_active'
                        );
                    """)
                    
                    if column_exists:
                        rankings = await pool.fetch("""
                            SELECT bot_id, rank_score, is_active
                            FROM bot_rankings
                            WHERE is_active = true
                            ORDER BY rank_score DESC;
                        """)
                    else:
                        # If is_active doesn't exist, assume all bots are active
                        rankings = await pool.fetch("""
                            SELECT bot_id, rank_score
                            FROM bot_rankings
                            ORDER BY rank_score DESC;
                        """)
                    
                    if not rankings:
                        return None
                        
                    # Calculate total score for all bots (active ones if the column exists)
                    if column_exists:
                        total_score = sum(row['rank_score'] for row in rankings if row['is_active'])
                    else:
                        total_score = sum(row['rank_score'] for row in rankings)
                    
                    allocations = []
                    for row in rankings:
                        # Skip inactive bots if is_active column exists
                        if column_exists and not row['is_active']:
                            continue
                            
                        allocation = (row['rank_score'] / total_score) * total_funds if total_score > 0 else total_funds / len(rankings)
                        allocations.append({
                            'bot_id': row['bot_id'],
                            'allocated_amount': allocation,
                            'percentage': (allocation / total_funds) * 100,
                            'rank_score': row['rank_score']
                        })
                    
                    return allocations
        except Exception as e:
            st.error(f"Error calculating fund allocation: {str(e)}")
            return None
    
    # Function to update variable weights
    async def update_weight(variable_name, weight):
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # First check if the variable exists
                var_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM variable_weights WHERE variable_name = $1
                    )
                """, variable_name)
                
                if var_exists:
                    # Update existing record
                    await pool.execute("""
                        UPDATE variable_weights
                        SET weight = $2, last_updated = NOW()
                        WHERE variable_name = $1
                    """, variable_name, weight)
                else:
                    # Insert new record
                    await pool.execute("""
                        INSERT INTO variable_weights (variable_name, weight, last_updated)
                        VALUES ($1, $2, NOW())
                    """, variable_name, weight)
                
                return True
        except Exception as e:
            st.error(f"Error updating weight: {str(e)}")
            return False
    
    with rank_tab1:
        st.subheader("Current Bot Rankings")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Refresh Rankings"):
                rankings = asyncio.run(fetch_bot_rankings())
                if rankings:
                    # Convert to DataFrame
                    rankings_df = pd.DataFrame([dict(r) for r in rankings])
                    
                    # Create visualization of current rankings
                    fig = go.Figure()
                    
                    # Get ticker and bot_id for labels
                    labels = [f"Bot {row['bot_id']} - {row['ticker']}" for i, row in rankings_df.iterrows()]
                    
                    # Create bar chart of rank scores
                    fig.add_trace(go.Bar(
                        x=labels,
                        y=rankings_df['rank_score'],
                        text=rankings_df['rank_score'].apply(lambda x: f"{x:.2f}"),
                        textposition='auto',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title='Bot Ranking Scores',
                        xaxis_title='Bot',
                        yaxis_title='Rank Score',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display rankings table
                    st.dataframe(
                        rankings_df[[
                            'bot_id', 'ticker', 'rank_score', 
                            'one_day_performance', 'avg_win_rate', 'profit_factor', 
                            'is_active', 'timestamp'
                        ]].style.format({
                            'rank_score': lambda x: safe_format(x, '{:.2f}'),
                            'one_day_performance': lambda x: safe_format(x, '{:.2f}%'),
                            'avg_win_rate': lambda x: safe_format(x, '{:.2f}%'),
                            'profit_factor': lambda x: safe_format(x, '{:.2f}'),
                            'timestamp': lambda x: safe_format(x, '{:%Y-%m-%d %H:%M:%S}')
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No ranking data available. Please make sure the bot_rankings table exists and contains data.")
        
        with col2:
            # Fund allocation
            st.subheader("Fund Allocation")
            
            # Input for total funds
            total_funds = st.number_input("Total Funds ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
            
            # Add allocation mode selection
            allocation_mode = st.radio(
                "Allocation Method",
                ["Proportional", "Top 10 Strategy"],
                help="Choose 'Proportional' to allocate funds based on rank scores, or 'Top 10 Strategy' to give $2000 to each top 10 bot"
            )
            
            if st.button("Calculate Allocation"):
                if allocation_mode == "Proportional":
                    # Use existing proportional allocation
                    allocations = asyncio.run(fetch_fund_allocation(total_funds))
                else:
                    # Use new Top 10 allocation
                    # First check if we can import the BotRanker
                    try:
                        # Add the parent directory to the path to find the src module
                        import sys
                        import os
                        # Get the current directory
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        # Go up two levels (from user_interface/src to the project root)
                        project_root = os.path.abspath(os.path.join(current_dir, '../..'))
                        # Add to path if not already there
                        if project_root not in sys.path:
                            sys.path.insert(0, project_root)
                        
                        # Now import the BotRanker
                        from src.bot_ranker import BotRanker
                        
                        # Create a DB pool
                        pool = asyncio.run(create_db_pool(**DB_CONFIG))
                        
                        # Create a bot ranker instance and get top 10 allocation
                        ranker = BotRanker(pool)
                        allocations = asyncio.run(ranker.get_top10_fund_allocation(2000))
                        
                        # Close the pool when done
                        asyncio.run(pool.close())
                    except ImportError as e:
                        st.error(f"Error importing BotRanker: {str(e)}")
                        allocations = None
                    
                    if allocations:
                        # Convert to DataFrame
                        alloc_df = pd.DataFrame(allocations)
                        
                        # Create visualization based on allocation method
                        if allocation_mode == "Proportional":
                            # Create pie chart of allocations for proportional allocation
                            fig = go.Figure(data=[go.Pie(
                                labels=[f"Bot {row['bot_id']}" for i, row in alloc_df.iterrows()],
                                values=alloc_df['allocated_amount'],
                                textinfo='label+percent',
                                hoverinfo='label+value+percent',
                                marker=dict(
                                    # Define colors based on rank for visual differentiation
                                    colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
                                )
                            )])
                            
                            fig.update_layout(
                                title=f'Proportional Fund Allocation (${total_funds:,})',
                                height=300
                            )
                        else:
                            # For Top 10 allocation, create a bar chart
                            # Sort by rank first
                            sorted_df = alloc_df.sort_values('rank')
                            
                            # Create labels with rank, bot ID and ticker where available
                            labels = []
                            for _, row in sorted_df.iterrows():
                                if 'ticker' in row and row['ticker']:
                                    label = f"Rank {row['rank']}: Bot {row['bot_id']} ({row['ticker']})"
                                else:
                                    label = f"Rank {row['rank']}: Bot {row['bot_id']}"
                                labels.append(label)
                            
                            # Create a categorical color scale: green for top 10, gray for others
                            colors = ['green' if row['top_10'] else 'lightgray' for _, row in sorted_df.iterrows()]
                            
                            fig = go.Figure(data=[go.Bar(
                                x=labels,
                                y=sorted_df['allocated_amount'],
                                text=sorted_df['allocated_amount'].apply(lambda x: f"${x:,.2f}"),
                                textposition='auto',
                                marker_color=colors
                            )])
                            
                            fig.update_layout(
                                title=f'Top 10 Fund Allocation (${len(sorted_df[sorted_df["top_10"]]) * 2000:,})',
                                xaxis_title='Bot',
                                yaxis_title='Allocation Amount ($)',
                                xaxis_tickangle=-45,
                                height=400
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display allocation table
                        alloc_df_display = alloc_df.copy()
                        if 'ticker' in alloc_df_display.columns:
                            alloc_df_display['bot'] = alloc_df_display.apply(
                                lambda x: f"Bot {x['bot_id']} - {x['ticker']}", axis=1
                            )
                        else:
                            alloc_df_display['bot'] = alloc_df_display['bot_id'].apply(lambda x: f"Bot {x}")
                        
                        columns_to_display = ['bot', 'rank', 'allocated_amount', 'percentage', 'rank_score']
                        if 'top_10' in alloc_df_display.columns:
                            columns_to_display.append('top_10')
                        
                        st.dataframe(
                            alloc_df_display[columns_to_display].style.format({
                                'allocated_amount': lambda x: safe_format(x, '${:.2f}'),
                                'percentage': lambda x: safe_format(x, '{:.2f}%'),
                                'rank_score': lambda x: safe_format(x, '{:.2f}')
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No allocation data available. Please make sure the bot_rankings table exists and contains data.")
            
            # Toggle bot active status
            st.subheader("Bot Status Management")
            
            async def fetch_bot_ids():
                try:
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        bot_ids = await pool.fetch("""
                            SELECT DISTINCT bot_id, is_active 
                            FROM bot_rankings
                            ORDER BY bot_id
                        """)
                        return bot_ids
                except Exception as e:
                    st.error(f"Error fetching bot IDs: {str(e)}")
                    return None
            
            bot_ids = asyncio.run(fetch_bot_ids())
            
            if bot_ids:
                bot_id = st.selectbox(
                    "Select Bot ID",
                    [row['bot_id'] for row in bot_ids]
                )
                
                # Find current status
                current_status = next((row['is_active'] for row in bot_ids if row['bot_id'] == bot_id), True)
                
                status = st.checkbox("Active", value=current_status)
                
                if st.button("Update Status"):
                    async def toggle_status():
                        try:
                            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                                # Try to import BotRanker to use its method
                                try:
                                    from src.bot_ranker import BotRanker
                                    ranker = BotRanker(pool)
                                    success = await ranker.toggle_bot_active_status(bot_id, status)
                                except ImportError:
                                    # Fallback: Update directly in database
                                    await pool.execute("""
                                        UPDATE bot_rankings
                                        SET is_active = $2
                                        WHERE bot_id = $1
                                    """, bot_id, status)
                                    success = True
                                return success
                        except Exception as e:
                            st.error(f"Error updating bot status: {str(e)}")
                            return False
                    
                    success = asyncio.run(toggle_status())
                    if success:
                        st.success(f"Bot {bot_id} status updated to {'Active' if status else 'Inactive'}")
            else:
                st.info("No bots found in the rankings table.")
    
    with rank_tab2:
        st.subheader("Historical Ranking Performance")
        
        # Date range selection
        days = st.slider("Days to Display", min_value=7, max_value=90, value=30, step=1)
        
        if st.button("Show Historical Rankings"):
            historical_rankings = asyncio.run(fetch_historical_rankings(days))
            if historical_rankings:
                # Convert to DataFrame
                hist_df = pd.DataFrame([dict(r) for r in historical_rankings])
                
                # Create line chart of rank position over time
                fig = go.Figure()
                
                for bot_id in hist_df['bot_id'].unique():
                    bot_data = hist_df[hist_df['bot_id'] == bot_id]
                    
                    # Get ticker for this bot if available
                    ticker = bot_data['ticker'].iloc[0] if 'ticker' in bot_data.columns and not bot_data['ticker'].empty else 'Unknown'
                    
                    fig.add_trace(go.Scatter(
                        x=bot_data['date'],
                        y=bot_data['rank_score'],
                        mode='lines+markers',
                        name=f"Bot {bot_id} - {ticker}",
                        hovertemplate='Date: %{x}<br>Score: %{y}'
                    ))
                
                # Do NOT invert y-axis since higher rank_score is better
                fig.update_layout(
                    title='Bot Ranking Score History',
                    xaxis_title='Date',
                    yaxis_title='Rank Score',
                    hovermode='closest',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Heat map visualization of rankings over time
                if len(hist_df['date'].unique()) > 1:
                    # Pivot the data for the heatmap
                    pivot_df = hist_df.pivot(index='bot_id', columns='date', values='rank_score')
                    
                    # Replace bot_id with bot_id - ticker
                    bot_labels = []
                    for bot_id in pivot_df.index:
                        ticker = hist_df[hist_df['bot_id'] == bot_id]['ticker'].iloc[0] if 'ticker' in hist_df.columns else 'Unknown'
                        bot_labels.append(f"Bot {bot_id} - {ticker}")
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=bot_labels,
                        colorscale='Viridis',
                        reversescale=True,  # Reverse so rank 1 (best) is dark color
                        zmin=1,
                        zmax=len(pivot_df.index)
                    ))
                    
                    fig.update_layout(
                        title='Bot Ranking Heatmap',
                        xaxis_title='Date',
                        yaxis_title='Bot',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical ranking data available for the selected time period.")
    
    # Function to check and update the variable_weights table schema
    async def ensure_variable_weights_schema():
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # Check if table exists
                table_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'variable_weights'
                    );
                """)
                
                if table_exists:
                    # Check if the unique constraint exists
                    constraint_exists = await pool.fetchval("""
                        SELECT COUNT(*) FROM information_schema.table_constraints 
                        WHERE table_name = 'variable_weights' 
                        AND constraint_type = 'UNIQUE'
                        AND constraint_name LIKE '%variable_name%';
                    """)
                    
                    if not constraint_exists:
                        # Add the unique constraint
                        try:
                            await pool.execute("""
                                ALTER TABLE variable_weights 
                                ADD CONSTRAINT variable_weights_variable_name_key UNIQUE (variable_name);
                            """)
                            st.success("Added missing UNIQUE constraint to variable_weights table.")
                        except Exception as e:
                            st.warning(f"Could not add UNIQUE constraint: {str(e)}")
                
                return True
        except Exception as e:
            st.error(f"Error checking variable_weights schema: {str(e)}")
            return False
    
    with rank_tab3:
        st.subheader("Weight Management")
        
        # Ensure the variable_weights table has the necessary constraints
        asyncio.run(ensure_variable_weights_schema())
        
        # Get the hardcoded weights for comparison
        async def get_hardcoded_weights():
            try:
                # Try to import the BotRanker to get its hardcoded weights
                try:
                    # Add the project root to the Python path
                    import sys
                    import os
                    
                    # Get the current directory and project root
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
                    
                    # Add to path if not already there
                    if project_root not in sys.path:
                        sys.path.append(project_root)
                    
                    # Now import the BotRanker
                    from src.bot_ranker import BotRanker
                    
                    # Create a temporary pool for initialization
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        ranker = BotRanker(pool)
                        return await ranker.get_variable_weights()
                except ImportError as e:
                    st.error(f"Error importing BotRanker: {str(e)}")
                    
                    # Return default weights if import fails
                    return {
                        'one_hour_performance': 15.0,
                        'two_hour_performance': 10.0,
                        'one_day_performance': 12.0,
                        'one_week_performance': 8.0,
                        'one_month_performance': 5.0,
                        'avg_win_rate': 12.0,
                        'profit_per_second': 10.0,
                        'total_pnl': 8.0,
                        'profit_factor': 8.0,
                        'avg_profit_per_trade': 6.0,
                        'avg_drawdown': -5.0,
                        'max_drawdown': -7.0,
                        'sharpe_ratio': 8.0,
                        'price_model_score': 9.0,
                        'volume_model_score': 7.0,
                        'price_wall_score': 6.0,
                        'win_streak_2': 3.0,
                        'win_streak_3': 4.0,
                        'win_streak_4': 5.0,
                        'win_streak_5': 6.0,
                    }
            except Exception as e:
                st.error(f"Error getting hardcoded weights: {str(e)}")
                return {}
        
        weights = asyncio.run(get_hardcoded_weights())
        
        if weights:
            # Convert weights to DataFrame for display
            weights_df = pd.DataFrame({
                'variable_name': list(weights.keys()),
                'weight': list(weights.values())
            })
            
            # Categorize variables for better organization
            categories = {
                'Performance Periods': [var for var in weights.keys() if 'performance' in var],
                'Core Metrics': ['avg_win_rate', 'profit_per_second', 'total_pnl', 
                                'profit_factor', 'avg_profit_per_trade'],
                'Risk Metrics': ['avg_drawdown', 'max_drawdown', 'sharpe_ratio'],
                'Model Scores': ['price_model_score', 'volume_model_score', 'price_wall_score'],
                'Win Streaks': [var for var in weights.keys() if 'win_streak' in var]
            }
            
            # Display weights by category
            for category, vars in categories.items():
                st.write(f"### {category}")
                
                category_weights = weights_df[weights_df['variable_name'].isin(vars)].copy()
                
                if not category_weights.empty:
                    # Add abs_weight for visualization (absolute value for negative weights)
                    category_weights['abs_weight'] = category_weights['weight'].abs()
                    
                    # Create horizontal bar chart
                    fig = go.Figure()
                    
                    # Add bars with different colors based on positive/negative weight
                    fig.add_trace(go.Bar(
                        x=category_weights['abs_weight'],
                        y=category_weights['variable_name'],
                        orientation='h',
                        text=category_weights['weight'].apply(lambda x: f"{x:.1f}"),
                        textposition='auto',
                        marker=dict(
                            color=category_weights['weight'].apply(
                                lambda x: 'rgb(26, 118, 255)' if x >= 0 else 'rgb(255, 50, 50)'
                            )
                        )
                    ))
                    
                    fig.update_layout(
                        xaxis_title='Weight Value (Absolute)',
                        yaxis_title='Variable Name',
                        height=max(50 + 30 * len(category_weights), 200),  # Dynamic height based on number of variables
                        margin=dict(l=10, r=10, t=20, b=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Allow editing weights in this category
                    for i, row in category_weights.iterrows():
                        var_name = row['variable_name']
                        current_weight = row['weight']
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            new_weight = st.slider(
                                f"Weight for {var_name}",
                                min_value=-20.0 if current_weight < 0 else 0.0,  # Allow negative only if already negative
                                max_value=20.0,
                                value=float(current_weight),
                                step=0.5,
                                key=f"weight_{var_name}"
                            )
                        with col2:
                            if st.button("Update", key=f"update_{var_name}"):
                                # Here we would ideally update the weights in the bot_ranker.py file
                                # Since that's not directly possible, we'll update the variable_weights table
                                st.warning("""
                                Note: This updates the weight in the database, but not in the hardcoded 
                                weights in bot_ranker.py. To make the change permanent, you'll need to 
                                manually update the get_variable_weights method in src/bot_ranker.py.
                                """)
                                
                                success = asyncio.run(update_weight(var_name, new_weight))
                                if success:
                                    st.success(f"Weight for {var_name} updated to {new_weight}")
                else:
                    st.info(f"No variables found for category: {category}")
        else:
            st.warning("Unable to retrieve ranking weights.")

    with rank_tab4:
        st.subheader("Ranking System Diagnostics")
        
        # Add explanation
        st.write("""
        This section provides direct access to the database tables related to the bot ranking system.
        Use these views to verify that the ranking system is working correctly.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Direct view of bot_rankings table
            st.write("### Bot Rankings Table")
            
            # Function to get raw database data
            async def fetch_raw_rankings():
                try:
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        # Check if table exists
                        table_exists = await pool.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'bot_rankings'
                            );
                        """)
                        
                        if not table_exists:
                            return None
                            
                        # Get all fields from the bot_rankings table
                        rankings = await pool.fetch("""
                            SELECT * FROM bot_rankings
                            ORDER BY rank_score DESC, timestamp DESC;
                        """)
                        return rankings
                except Exception as e:
                    st.error(f"Error fetching raw rankings data: {str(e)}")
                    return None
            
            if st.button("View Bot Rankings Table"):
                raw_rankings = asyncio.run(fetch_raw_rankings())
                if raw_rankings:
                    # Convert to DataFrame
                    df = pd.DataFrame([dict(r) for r in raw_rankings])
                    st.dataframe(
                        df.style.format({
                            'rank_score': lambda x: safe_format(x, '{:.2f}'),
                            'timestamp': lambda x: safe_format(x, '{:%Y-%m-%d %H:%M:%S}')
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No data found in bot_rankings table or table doesn't exist.")
        
        with col2:
            # Variable weights table view
            st.write("### Variable Weights Table")
            
            async def fetch_variable_weights_raw():
                try:
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        # Check if table exists
                        table_exists = await pool.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'variable_weights'
                            );
                        """)
                        
                        if not table_exists:
                            return None
                            
                        # Get all fields from the variable_weights table
                        weights = await pool.fetch("""
                            SELECT * FROM variable_weights
                            ORDER BY variable_name;
                        """)
                        return weights
                except Exception as e:
                    st.error(f"Error fetching variable weights data: {str(e)}")
                    return None
            
            if st.button("View Variable Weights Table"):
                variable_weights = asyncio.run(fetch_variable_weights_raw())
                if variable_weights:
                    # Convert to DataFrame
                    df = pd.DataFrame([dict(r) for r in variable_weights])
                    st.dataframe(
                        df.style.format({
                            'weight': lambda x: safe_format(x, '{:.2f}'),
                            'last_updated': lambda x: safe_format(x, '{:%Y-%m-%d %H:%M:%S}')
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No data found in variable_weights table or table doesn't exist.")
        
        # Add a divider
        st.markdown("---")
        
        # Manual ranking recalculation for testing
        st.write("### Manual Ranking Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Recalculate Bot Rankings")
            st.write("This will trigger the ranking calculation process and update the database.")
            
            if st.button("Recalculate Rankings"):
                try:
                    async def run_ranking():
                        try:
                            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                                # Add the parent directory to the path to find the src module
                                import sys
                                import os
                                # Get the current directory
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                # Go up two levels (from user_interface/src to the project root)
                                project_root = os.path.abspath(os.path.join(current_dir, '../..'))
                                # Add to path if not already there
                                if project_root not in sys.path:
                                    sys.path.insert(0, project_root)
                                
                                # Now import the BotRanker
                                from src.bot_ranker import BotRanker
                                ranker = BotRanker(pool)
                                
                                # Log start time
                                start_time = time.time()
                                
                                # Run the ranking process
                                ranked_bots = await ranker.rank_bots()
                                
                                # Log completion time
                                end_time = time.time()
                                
                                return {
                                    "success": True,
                                    "bots_ranked": len(ranked_bots),
                                    "time_taken": end_time - start_time
                                }
                        except Exception as e:
                            st.error(f"Error in ranking process: {str(e)}")
                            return {
                                "success": False,
                                "error": str(e)
                            }
                    
                    result = asyncio.run(run_ranking())
                    
                    if result["success"]:
                        st.success(f"Successfully ranked {result['bots_ranked']} bots in {result['time_taken']:.2f} seconds!")
                    else:
                        st.error("Failed to recalculate rankings.")
                        
                except Exception as e:
                    st.error(f"Error recalculating rankings: {str(e)}")
        
        with col2:
            st.write("View Bot Metrics Table")
            st.write("This shows the raw metrics used for ranking calculation.")
            
            async def fetch_bot_metrics_raw():
                try:
                    async with asyncpg.create_pool(**DB_CONFIG) as pool:
                        # Check if table exists
                        table_exists = await pool.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'bot_metrics'
                            );
                        """)
                        
                        if not table_exists:
                            return None
                            
                        # Get the most recent entry for each bot
                        metrics = await pool.fetch("""
                            SELECT DISTINCT ON (bot_id) *
                            FROM bot_metrics
                            ORDER BY bot_id, timestamp DESC;
                        """)
                        return metrics
                except Exception as e:
                    st.error(f"Error fetching bot metrics: {str(e)}")
                    return None
            
            if st.button("View Recent Bot Metrics"):
                bot_metrics = asyncio.run(fetch_bot_metrics_raw())
                if bot_metrics:
                    # Convert to DataFrame
                    df = pd.DataFrame([dict(r) for r in bot_metrics])
                    
                    # Select only key columns to display
                    display_columns = ['bot_id', 'ticker', 'algo_id', 'timestamp']
                    metric_columns = [col for col in df.columns if col not in ['bot_id', 'ticker', 'algo_id', 'timestamp', 'last_updated']]
                    
                    # Sort metric columns alphabetically for better readability
                    metric_columns.sort()
                    
                    # Combine columns for display
                    display_df = df[display_columns + metric_columns]
                    
                    # Display in an expandable section due to many columns
                    with st.expander("Bot Metrics Data (Click to expand)"):
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No data found in bot_metrics table or table doesn't exist.")
        
        # Add information about the hardcoded weights
        st.markdown("---")
        st.write("### Current Hardcoded Weights")
        st.write("""
        These are the weights defined in the `get_variable_weights` method of the `BotRanker` class.
        To modify these permanently, you need to edit the source code in `src/bot_ranker.py`.
        """)
        
        # Get and display current hardcoded weights
        hardcoded_weights = asyncio.run(get_hardcoded_weights())
        if hardcoded_weights:
            # Create a nicely formatted table of the weights
            weights_list = [{"Metric": k, "Weight": v} for k, v in hardcoded_weights.items()]
            weights_df = pd.DataFrame(weights_list)
            
            # Sort by absolute weight value (descending)
            weights_df['Abs_Weight'] = weights_df['Weight'].abs()
            weights_df = weights_df.sort_values('Abs_Weight', ascending=False).drop('Abs_Weight', axis=1)
            
            st.dataframe(weights_df, use_container_width=True)

    # New function to check and update bot funding status
    async def check_and_update_bot_funding(check_interval_minutes=60):
        """
        Periodically check bot rankings and update fund allocations.
        
        This function:
        1. Checks the current top 10 ranked bots
        2. Allocates $2000 to each top 10 bot
        3. Removes funding from bots that fall out of the top 10
        4. Logs all fund allocation changes
        5. Updates the database with the new allocations
        
        Args:
            check_interval_minutes: How often to check rankings (in minutes)
        
        Returns:
            Dictionary with update status and changes made
        """
        try:
            # Import BotRanker
            try:
                # Add the parent directory to the path to find the src module
                import sys
                import os
                # Get the current directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up two levels (from user_interface/src to the project root)
                project_root = os.path.abspath(os.path.join(current_dir, '../..'))
                # Add to path if not already there
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # Now import the BotRanker
                from src.bot_ranker import BotRanker
            except ImportError as e:
                return {"success": False, "error": f"Error importing BotRanker: {str(e)}"}
            
            # Create a new DB pool
            pool = await create_db_pool(**DB_CONFIG)
            
            # Create a bot ranker instance
            ranker = BotRanker(pool)
            
            # Get the top 10 allocation
            allocations = await ranker.get_top10_fund_allocation(2000)
            
            # Check if table exists, create if not
            table_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'bot_fund_allocations'
                );
            """)
            
            if not table_exists:
                # Create the fund allocations table
                await pool.execute("""
                    CREATE TABLE bot_fund_allocations (
                        allocation_id SERIAL PRIMARY KEY,
                        bot_id INTEGER NOT NULL,
                        allocation_amount NUMERIC(10, 2) NOT NULL,
                        is_top_10 BOOLEAN NOT NULL,
                        previous_amount NUMERIC(10, 2),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        allocation_notes TEXT
                    );
                """)
            
            # Get current allocations from database
            current_allocations = await pool.fetch("""
                SELECT DISTINCT ON (bot_id) 
                    bot_id, allocation_amount, is_top_10, timestamp 
                FROM bot_fund_allocations
                ORDER BY bot_id, timestamp DESC;
            """)
            
            # Convert to dictionary for easy lookup
            current_alloc_map = {row['bot_id']: dict(row) for row in current_allocations}
            
            # Prepare changes to record
            changes = []
            
            # Update allocations in database
            for bot in allocations:
                bot_id = bot['bot_id']
                amount = bot['allocation_amount']
                is_top_10 = bot['top_10']
                
                # Check if this bot's allocation has changed
                if bot_id in current_alloc_map:
                    prev = current_alloc_map[bot_id]
                    prev_amount = prev['allocation_amount']
                    prev_top_10 = prev['is_top_10']
                    
                    # Record changes for bots entering or leaving top 10
                    if prev_top_10 != is_top_10:
                        note = ""
                        if is_top_10 and not prev_top_10:
                            note = f"Bot {bot_id} entered top 10, allocated ${amount}"
                            changes.append(f"✅ {note}")
                        elif not is_top_10 and prev_top_10:
                            note = f"Bot {bot_id} fell out of top 10, funding removed"
                            changes.append(f"❌ {note}")
                        
                        # Insert allocation change record
                        await pool.execute("""
                            INSERT INTO bot_fund_allocations 
                                (bot_id, allocation_amount, is_top_10, previous_amount, allocation_notes)
                            VALUES ($1, $2, $3, $4, $5)
                        """, bot_id, amount, is_top_10, prev_amount, note)
                else:
                    # New bot, no previous allocation
                    note = ""
                    if is_top_10:
                        note = f"Initial allocation for bot {bot_id}: ${amount}"
                        changes.append(f"✅ {note}")
                    else:
                        note = f"Bot {bot_id} not in top 10, no funds allocated"
                    
                    # Insert new allocation record
                    await pool.execute("""
                        INSERT INTO bot_fund_allocations 
                            (bot_id, allocation_amount, is_top_10, allocation_notes)
                        VALUES ($1, $2, $3, $4)
                    """, bot_id, amount, is_top_10, note)
            
            # Close the pool
            await pool.close()
            
            # Create result object
            result = {
                "success": True,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "changes": changes,
                "top_10_bots": [a for a in allocations if a['top_10']],
                "total_allocated": sum(a['allocation_amount'] for a in allocations if a['top_10']),
                "allocation_count": len([a for a in allocations if a['top_10']])
            }
            
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Function to fetch allocation history
    async def fetch_allocation_history(limit=50):
        """Fetch recent allocation changes from the database"""
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # Check if table exists
                table_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'bot_fund_allocations'
                    );
                """)
                
                if not table_exists:
                    return None
                
                # Fetch allocation history with bot information
                history = await pool.fetch("""
                    SELECT 
                        a.allocation_id, a.bot_id, a.allocation_amount, a.is_top_10, 
                        a.previous_amount, a.timestamp, a.allocation_notes,
                        bm.ticker
                    FROM bot_fund_allocations a
                    LEFT JOIN (
                        SELECT DISTINCT ON (bot_id) bot_id, ticker
                        FROM bot_metrics
                        ORDER BY bot_id, timestamp DESC
                    ) bm ON a.bot_id = bm.bot_id
                    ORDER BY a.timestamp DESC
                    LIMIT $1
                """, limit)
                
                return history
        except Exception as e:
            st.error(f"Error fetching allocation history: {str(e)}")
            return None
    
    # Add content to rank_tab5 for automated fund management
    with rank_tab5:
        st.subheader("Automated Fund Management")
        
        st.markdown("""
        This section provides automated management of fund allocations based on bot rankings.
        The system periodically checks the current rankings and allocates funds as follows:
        
        - Each of the **top 10 ranked bots** is allocated **$2,000**
        - Bots that fall out of the top 10 have their funds removed
        - Bots that enter the top 10 are allocated $2,000
        """)
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show allocation history
            st.subheader("Allocation History")
            
            history_limit = st.slider("Number of records to show", 10, 100, 30, 5)
            
            if st.button("View Allocation History"):
                history = asyncio.run(fetch_allocation_history(history_limit))
                if history:
                    # Convert to DataFrame
                    history_df = pd.DataFrame([dict(r) for r in history])
                    
                    # Format for display
                    history_df['bot'] = history_df.apply(
                        lambda x: f"Bot {x['bot_id']} ({x['ticker']})" if pd.notnull(x['ticker']) else f"Bot {x['bot_id']}",
                        axis=1
                    )
                    
                    # Add readable label for top 10 status
                    history_df['status'] = history_df['is_top_10'].apply(
                        lambda x: "✅ In Top 10" if x else "❌ Not in Top 10"
                    )
                    
                    # Add change amount column
                    history_df['change'] = history_df['allocation_amount'] - history_df['previous_amount']
                    
                    # Display the DataFrame
                    st.dataframe(
                        history_df[[
                            'bot', 'timestamp', 'allocation_amount', 
                            'previous_amount', 'change', 'status', 'allocation_notes'
                        ]].style.format({
                            'allocation_amount': lambda x: safe_format(x, '${:.2f}'),
                            'previous_amount': lambda x: safe_format(x, '${:.2f}'),
                            'change': lambda x: safe_format(x, '${:.2f}'),
                            'timestamp': lambda x: safe_format(x, '{:%Y-%m-%d %H:%M:%S}')
                        }),
                        use_container_width=True
                    )
                    
                    # Create a line chart showing allocation changes over time
                    pivot_df = history_df.pivot_table(
                        index='timestamp', 
                        columns='bot', 
                        values='allocation_amount',
                        aggfunc='first'
                    ).reset_index()
                    
                    if not pivot_df.empty and len(pivot_df.columns) > 1:
                        fig = go.Figure()
                        
                        for col in pivot_df.columns:
                            if col != 'timestamp':
                                fig.add_trace(go.Scatter(
                                    x=pivot_df['timestamp'],
                                    y=pivot_df[col],
                                    mode='lines+markers',
                                    name=col
                                ))
                        
                        fig.update_layout(
                            title='Bot Allocation History',
                            xaxis_title='Time',
                            yaxis_title='Allocation Amount ($)',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No allocation history available yet.")
        
        with col2:
            # Manual update control
            st.subheader("Update Allocations")
            
            update_interval = st.number_input(
                "Check Interval (minutes)",
                min_value=1,
                max_value=1440,  # Max 24 hours
                value=60,
                help="How often the system should check rankings and update allocations"
            )
            
            if st.button("Update Fund Allocations Now"):
                with st.spinner("Updating fund allocations..."):
                    result = asyncio.run(check_and_update_bot_funding(update_interval))
                
                if result["success"]:
                    st.success("Fund allocations updated successfully!")
                    
                    # Show summary of changes
                    if result["changes"]:
                        st.subheader("Changes Made:")
                        for change in result["changes"]:
                            st.write(change)
                    else:
                        st.info("No changes were needed to the allocations.")
                    
                    # Show totals
                    st.metric(
                        "Total Allocated",
                        f"${result['total_allocated']:,.2f}",
                        f"{result['allocation_count']} bots"
                    )
                else:
                    st.error(f"Error updating allocations: {result.get('error', 'Unknown error')}")
            
            # Schedule automated updates
            st.subheader("Automated Updates")
            
            # This would normally use a scheduled task, but for demo we'll show a placeholder
            st.info("""
            For production use, you should set up a scheduled task to call the 
            `check_and_update_bot_funding()` function at your desired interval.
            
            In a production environment, this could be implemented with:
            - A cron job on Linux/Unix systems
            - Windows Task Scheduler
            - A cloud function triggered on a schedule
            - A dedicated scheduler within your application
            """)
                        
            # Show current top 10
            st.subheader("Current Top 10 Bots")
            
            # Get current top 10 allocations
            current_top10 = asyncio.run(fetch_allocation_history(10))
            if current_top10:
                top10_df = pd.DataFrame([dict(r) for r in current_top10])
                top10_df = top10_df[top10_df['is_top_10'] == True]
                
                if not top10_df.empty:
                    # Format for display
                    top10_df['bot'] = top10_df.apply(
                        lambda x: f"Bot {x['bot_id']} ({x['ticker']})" if pd.notnull(x['ticker']) else f"Bot {x['bot_id']}",
                        axis=1
                    )
                    
                    st.dataframe(
                        top10_df[['bot', 'allocation_amount']].style.format({
                            'allocation_amount': lambda x: safe_format(x, '${:.2f}')
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No bots are currently in the top 10.")
            else:
                st.info("No allocation data available yet.")

def trade_analysis():
    st.header("Trading Analytics Dashboard")

    db_params = {
        'dbname': 'tick_data',
        'user': 'clayb',
        'password': 'musicman',
        'host': 'localhost',
        'port': 5432
    }

    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            # 1) Bot Metrics: get the most recent row per (bot_id, ticker)
            cur.execute("""
                SELECT DISTINCT ON (bot_id, ticker)
                    bot_id,
                    ticker,
                    updated_at,
                    one_hour_performance,
                    one_day_performance,
                    avg_win_rate,
                    profit_per_second
                FROM bot_metrics
                WHERE bot_id BETWEEN 1 AND 8
                ORDER BY bot_id, ticker, updated_at DESC
            """)
            metrics_data = cur.fetchall()

            # 2) Trading Statistics: group by (bot_id, ticker),
            #    also TRIM(b.ticker) to avoid accidental duplicates if trailing spaces exist
            cur.execute("""
                SELECT 
                    b.bot_id,
                    TRIM(b.ticker) AS ticker,
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) AS profitable_trades,
                    AVG(trade_pnl) AS avg_pnl,
                    AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))) AS avg_duration
                FROM sim_bot_trades b
                WHERE b.bot_id BETWEEN 1 AND 8
                GROUP BY b.bot_id, TRIM(b.ticker)
                ORDER BY b.bot_id;
            """)
            stats_data = cur.fetchall()

    # Display Bot Metrics
    st.subheader("Current Bot Performance Metrics")
    # If no rows, metrics_data is empty
    if not metrics_data:
        st.info("No metrics found in bot_metrics table for bots 1-8.")
    else:
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=['Bot ID', 'Ticker', 'Timestamp', '1hr Perf', '24hr Perf', 'Win Rate', 'Profit/sec']
        )
        st.dataframe(
            metrics_df.style.format({
                '1hr Perf': lambda x: safe_format(x, '{:.2f}%'),
                '24hr Perf': lambda x: safe_format(x, '{:.2f}%'),
                'Win Rate': lambda x: safe_format(x, '{:.1f}%'),
                'Profit/sec': lambda x: safe_format(x, '${:.4f}')
            }),
            use_container_width=True
        )

    # Display Trading Statistics
    st.subheader("Aggregate Trading Statistics")
    if not stats_data:
        st.info("No trades found for bots 1-8 in sim_bot_trades table.")
    else:
        stats_df = pd.DataFrame(stats_data, columns=[
            'Bot ID', 'Ticker', 'Total Trades',
            'Profitable Trades', 'Avg PNL', 'Avg Duration (sec)'
        ])
        st.dataframe(
            stats_df.style.format({
                'Avg PNL': lambda x: safe_format(x, '${:.2f}'),
                'Avg Duration (sec)': lambda x: safe_format(x, '{:.1f}')
            }),
            use_container_width=True
        )

    # Today's Trading Statistics Section
    st.subheader("Today's Trading Statistics")
    
    with st.container():
        col1, col2 = st.columns(2)
        show_today = col1.button("Show Today's Trades")
        export_today = col2.button("Export Today's Trades")
        
        if show_today or export_today:
            async def fetch_todays_trades():
                async with asyncpg.create_pool(**DB_CONFIG) as pool:
                    # Use CURRENT_DATE to filter trades from today (based on entry_time)
                    sql = "SELECT * FROM sim_bot_trades WHERE entry_time::date = CURRENT_DATE;"
                    result = await pool.fetch(sql)
                    return [dict(r) for r in result]
            
            todays_trades = asyncio.run(fetch_todays_trades())
            if todays_trades:
                todays_df = pd.DataFrame(todays_trades)
                st.markdown("### Today's Trades Table")
                st.dataframe(todays_df)
                
                # Detailed Summary Calculation
                total_trades = todays_df.shape[0]
                total_pnl = todays_df['trade_pnl'].sum() if 'trade_pnl' in todays_df.columns else 0
                avg_pnl = todays_df['trade_pnl'].mean() if 'trade_pnl' in todays_df.columns else 0
                
                # Convert datetime columns in case they are not already datetime objects
                if 'entry_time' in todays_df.columns and 'exit_time' in todays_df.columns:
                    todays_df['entry_time'] = pd.to_datetime(todays_df['entry_time'], errors='coerce')
                    todays_df['exit_time'] = pd.to_datetime(todays_df['exit_time'], errors='coerce')
                    valid_durations = (todays_df['exit_time'] - todays_df['entry_time']).dt.total_seconds().dropna()
                    avg_duration = valid_durations.mean() if not valid_durations.empty else 0
                else:
                    avg_duration = 0
                
                st.markdown("### Today's Trading Summary")
                st.write("Total Trades:", total_trades)
                st.write("Total PNL: $", f"{total_pnl:.2f}")
                st.write("Average PNL: $", f"{avg_pnl:.2f}")
                st.write("Average Trade Duration (sec):", f"{avg_duration:.1f} seconds")
                
                if export_today:
                    csv_data = todays_df.to_csv(index=False)
                    st.download_button(
                        label="Download Today's Trades CSV",
                        data=csv_data,
                        file_name="todays_trades.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No trades found for today.")

    # Single unified Bot Metrics section
    st.subheader("Bot Performance Metrics")
    try:
        # Debug: Show connection info
        st.write("Attempting database connection...")
        
        async def fetch_bot_metrics():
            try:
                async with asyncpg.create_pool(**DB_CONFIG) as pool:
                    # Debug: Check if we can query the table
                    table_check = await pool.fetch("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'bot_metrics'
                        );
                    """)
                    st.write(f"Bot metrics table exists: {table_check[0]['exists']}")
                    
                    # Debug: Count rows
                    count = await pool.fetchval("SELECT COUNT(*) FROM bot_metrics;")
                    st.write(f"Number of rows in bot_metrics: {count}")
                    
                    if count > 0:
                        # Show sample of data
                        metrics = await pool.fetch("""
                            SELECT 
                                bot_id,
                                ticker,
                                one_hour_performance,
                                one_day_performance,
                                avg_win_rate,
                                profit_per_second,
                                last_updated
                            FROM bot_metrics
                            ORDER BY bot_id;
                        """)
                        return [dict(m) for m in metrics]
                    return None
            except Exception as e:
                st.error(f"Error accessing bot metrics: {str(e)}")
                st.write("Full error details:", e)

        metrics = asyncio.run(fetch_bot_metrics())
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(
                metrics_df.style.format({
                    'one_hour_performance': lambda x: safe_format(x, '{:.2f}%'),
                    'one_day_performance': lambda x: safe_format(x, '{:.2f}%'),
                    'avg_win_rate': lambda x: safe_format(x, '{:.1f}%'),
                    'profit_per_second': lambda x: safe_format(x, '${:.4f}')
                }),
                use_container_width=True
            )
        else:
            st.info("No bot metrics available in database")

    except Exception as e:
        st.error(f"Error accessing bot metrics: {str(e)}")
        st.write("Full error details:", e)

    # Display Trading Statistics
    st.subheader("Aggregate Trading Statistics")
    if not stats_data:
        st.info("No trades found for bots 1-8 in sim_bot_trades table.")
    else:
        stats_df = pd.DataFrame(stats_data, columns=[
            'Bot ID', 'Ticker', 'Total Trades',
            'Profitable Trades', 'Avg PNL', 'Avg Duration (sec)'
        ])
        st.dataframe(
            stats_df.style.format({
                'Avg PNL': lambda x: safe_format(x, '${:.2f}'),
                'Avg Duration (sec)': lambda x: safe_format(x, '{:.1f}')
            }),
            use_container_width=True
        ) 

    async def main():
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scheduled_tasks.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create a database pool
        pool = await asyncpg.create_pool(**DB_CONFIG)
        
        try:
            # Create an AI weight adjuster
            adjuster = AIWeightAdjuster(pool)
            
            # Run the adjuster once
            success = await adjuster.adjust_weights()
            if success:
                logging.info("Successfully adjusted weights")
            else:
                logging.error("Failed to adjust weights")
            
            # Run scheduled adjustment (runs in a continuous loop)
            # await adjuster.run_scheduled_adjustment(hours=24)
        finally:
            await pool.close()

    if __name__ == "__main__":
        asyncio.run(main())

# Function to check database schema
async def check_database_schema():
    try:
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Check for bot_metrics table
            bot_metrics_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'bot_metrics'
                );
            """)
            
            if bot_metrics_exists:
                st.success("✅ bot_metrics table exists")
            else:
                st.error("❌ bot_metrics table does not exist")
                
            # Check for bot_rankings table
            bot_rankings_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'bot_rankings'
                );
            """)
            
            if bot_rankings_exists:
                st.success("✅ bot_rankings table exists")
            else:
                st.error("❌ bot_rankings table does not exist")
                
            # Check for variable_weights table
            variable_weights_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'variable_weights'
                );
            """)
            
            if variable_weights_exists:
                st.success("✅ variable_weights table exists")
            else:
                st.error("❌ variable_weights table does not exist")
    except Exception as e:
        st.error(f"Error checking database schema: {str(e)}")

if st.button("Check Database Schema"):
    asyncio.run(check_database_schema())

# Add function to load logs from database
async def load_logs_from_db(bot_name=None, limit=1000):
    """
    Load logs from database
    
    Args:
        bot_name (str, optional): Name of the bot to load logs for. If None, load all logs.
        limit (int, optional): Maximum number of logs to load. Defaults to 1000.
        
    Returns:
        dict: Dictionary of logs by bot name
    """
    try:
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Check if logs table exists
            table_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'bot_logs'
                );
            """)
            
            if not table_exists:
                return {}
            
            # Query to get logs
            if bot_name:
                logs = await pool.fetch("""
                    SELECT bot_name, log_text, timestamp
                    FROM bot_logs
                    WHERE bot_name = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, bot_name, limit)
            else:
                logs = await pool.fetch("""
                    SELECT bot_name, log_text, timestamp
                    FROM bot_logs
                    ORDER BY timestamp DESC
                    LIMIT $1
                """, limit)
            
            # Organize logs by bot
            result = {}
            for log in logs:
                bot = log['bot_name']
                if bot not in result:
                    result[bot] = []
                    
                # Split log text into individual log lines
                log_lines = log['log_text'].splitlines()
                timestamp = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Add timestamp to each log line if missing
                formatted_lines = []
                for line in log_lines:
                    if line and not (line.startswith("20") or line.startswith("202")):
                        line = f"{timestamp} {line}"
                    formatted_lines.append(line)
                
                result[bot].extend(formatted_lines)
            
            # Sort logs chronologically
            for bot in result:
                result[bot].sort()
            
            return result
    except Exception as e:
        st.error(f"Error loading logs from database: {e}")
        return {}

# New function to fetch account details from Interactive Brokers
async def fetch_ib_account_details():
    """
    Fetch account details from Interactive Brokers.
    
    Returns:
        Dictionary with account summary data
    """
    try:
        # Import IB API
        import sys
        import os
        
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels (from user_interface/src to the project root)
        project_root = os.path.abspath(os.path.join(current_dir, '../..'))
        # Add to path if not already there
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            # Try to import the IB API module
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
            from ibapi.account_summary_tags import AccountSummaryTags
            from ibapi.contract import Contract
        except ImportError:
            return {
                "success": False,
                "error": "IB API not found. Please make sure ibapi is installed.",
                "sample_data": True  # Flag to use sample data
            }
        
        # Create a simple IB API client to fetch account info
        class AccountInfoClient(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)
                self.account_data = {}
                self.positions = []
                self.account_updates = []
                self.request_id = 1
                self.account_ready = False
                self.positions_ready = False
            
            def accountSummary(self, reqId, account, tag, value, currency):
                # Store account summary data
                if account not in self.account_data:
                    self.account_data[account] = {}
                self.account_data[account][tag] = {
                    "value": value,
                    "currency": currency
                }
            
            def accountSummaryEnd(self, reqId):
                self.account_ready = True
            
            def position(self, account, contract, position, avgCost):
                # Store position data
                self.positions.append({
                    "account": account,
                    "symbol": contract.symbol,
                    "exchange": contract.exchange,
                    "position": position,
                    "avgCost": avgCost
                })
            
            def positionEnd(self):
                self.positions_ready = True
            
            def updateAccountValue(self, key, value, currency, accountName):
                # Store real-time account updates
                self.account_updates.append({
                    "key": key,
                    "value": value,
                    "currency": currency,
                    "account": accountName,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Connect to IB and fetch data
        client = AccountInfoClient()
        client.connect("127.0.0.1", 7496, 0)  # Default IB Gateway port
        
        # Use a separate thread to process IB messages
        import threading
        api_thread = threading.Thread(target=client.run)
        api_thread.start()
        
        # Wait for connection
        max_wait = 5  # seconds
        wait_step = 0.1
        total_wait = 0
        
        while not client.isConnected() and total_wait < max_wait:
            time.sleep(wait_step)
            total_wait += wait_step
        
        if not client.isConnected():
            client.disconnect()
            return {
                "success": False,
                "error": "Could not connect to IB Gateway. Make sure it's running.",
                "sample_data": True  # Flag to use sample data
            }
        
        # Request account summary
        client.reqAccountSummary(client.request_id, "All", AccountSummaryTags.AllTags)
        client.request_id += 1
        
        # Request positions
        client.reqPositions()
        
        # Wait for data
        max_wait = 10  # seconds
        total_wait = 0
        
        while (not client.account_ready or not client.positions_ready) and total_wait < max_wait:
            time.sleep(wait_step)
            total_wait += wait_step
        
        # Disconnect from IB
        client.disconnect()
        
        # Format the response
        result = {
            "success": True,
            "account_data": client.account_data,
            "positions": client.positions,
            "account_updates": client.account_updates,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sample_data": True  # Flag to use sample data
        }

# Function to get sample account data when IB connection fails
def get_sample_account_data():
    """Provide sample account data for UI testing when IB connection is unavailable"""
    return {
        "account_data": {
            "DU12345": {
                "NetLiquidation": {"value": "52430.75", "currency": "USD"},
                "TotalCashValue": {"value": "15670.22", "currency": "USD"},
                "SettledCash": {"value": "15670.22", "currency": "USD"},
                "AccruedCash": {"value": "0.00", "currency": "USD"},
                "BuyingPower": {"value": "104861.50", "currency": "USD"},
                "EquityWithLoanValue": {"value": "52430.75", "currency": "USD"},
                "PreviousDayEquityWithLoanValue": {"value": "51982.50", "currency": "USD"},
                "GrossPositionValue": {"value": "36760.53", "currency": "USD"},
                "RegTEquity": {"value": "15670.22", "currency": "USD"},
                "RegTMargin": {"value": "18380.27", "currency": "USD"},
                "UnrealizedPnL": {"value": "1245.30", "currency": "USD"},
                "RealizedPnL": {"value": "450.75", "currency": "USD"},
                "ExchangeRate": {"value": "1.00", "currency": "USD"},
                "FundValue": {"value": "0.00", "currency": "USD"},
                "FullInitMarginReq": {"value": "7352.11", "currency": "USD"},
                "FullMaintMarginReq": {"value": "5881.68", "currency": "USD"},
                "FullAvailableFunds": {"value": "45078.64", "currency": "USD"},
                "DayTradesRemaining": {"value": "3", "currency": "USD"},
                "Leverage": {"value": "0.70", "currency": "USD"},
                "EquityPercentage": {"value": "100.00", "currency": "USD"}
            }
        },
        "positions": [
            {
                "account": "DU12345",
                "symbol": "COIN",
                "exchange": "NASDAQ",
                "position": 25,
                "avgCost": 175.25
            },
            {
                "account": "DU12345",
                "symbol": "TSLA",
                "exchange": "NASDAQ",
                "position": 15,
                "avgCost": 880.42
            },
            {
                "account": "DU12345",
                "symbol": "AAPL",
                "exchange": "NASDAQ",
                "position": 30,
                "avgCost": 145.70
            },
            {
                "account": "DU12345",
                "symbol": "SPY",
                "exchange": "ARCA",
                "position": -5,
                "avgCost": 420.35
            }
        ],
        "account_updates": [
            {
                "key": "NetLiquidation",
                "value": "52430.75",
                "currency": "USD",
                "account": "DU12345",
                "timestamp": (datetime.now() - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "key": "NetLiquidation",
                "value": "52385.25", 
                "currency": "USD",
                "account": "DU12345",
                "timestamp": (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "key": "NetLiquidation",
                "value": "52415.50",
                "currency": "USD",
                "account": "DU12345",
                "timestamp": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
            }
        ],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Function to store account history in the database
async def store_account_history(account_data):
    """
    Store account snapshot in the database for historical tracking
    
    Args:
        account_data: Dictionary containing account information
    """
    try:
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Check if table exists
            table_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'account_history'
                );
            """)
            
            if not table_exists:
                # Create the account history table
                await pool.execute("""
                    CREATE TABLE account_history (
                        history_id SERIAL PRIMARY KEY,
                        account_id TEXT NOT NULL,
                        net_liquidation NUMERIC(15, 2),
                        total_cash NUMERIC(15, 2),
                        buying_power NUMERIC(15, 2),
                        equity_with_loan NUMERIC(15, 2),
                        unrealized_pnl NUMERIC(15, 2),
                        realized_pnl NUMERIC(15, 2),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            
            # Extract account data
            for account_id, details in account_data.get("account_data", {}).items():
                # Store account snapshot
                await pool.execute("""
                    INSERT INTO account_history 
                        (account_id, net_liquidation, total_cash, buying_power, 
                        equity_with_loan, unrealized_pnl, realized_pnl)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                    account_id,
                    float(details.get("NetLiquidation", {}).get("value", 0)),
                    float(details.get("TotalCashValue", {}).get("value", 0)),
                    float(details.get("BuyingPower", {}).get("value", 0)),
                    float(details.get("EquityWithLoanValue", {}).get("value", 0)),
                    float(details.get("UnrealizedPnL", {}).get("value", 0)),
                    float(details.get("RealizedPnL", {}).get("value", 0))
                )
                
            return True
    except Exception as e:
        st.error(f"Error storing account history: {str(e)}")
        return False

# Function to fetch account history
async def fetch_account_history(days=30):
    """
    Fetch account history from the database
    
    Args:
        days: Number of days of history to fetch
        
    Returns:
        List of account history records
    """
    try:
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Check if table exists
            table_exists = await pool.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'account_history'
                );
            """)
            
            if not table_exists:
                return None
            
            # Fetch account history
            history = await pool.fetch("""
                SELECT * FROM account_history
                WHERE timestamp >= CURRENT_DATE - INTERVAL '$1 days'
                ORDER BY timestamp
            """, days)
            
            return history
    except Exception as e:
        st.error(f"Error fetching account history: {str(e)}")
        return None

# Account Information Tab
with tab_account:
    st.header("Interactive Brokers Account Information")
    
    st.markdown("""
    This section displays real-time information from your Interactive Brokers account.
    The data includes account balances, positions, and historical account value.
    """)
    
    # Create tabs for different account views
    acc_tab1, acc_tab2, acc_tab3 = st.tabs([
        "Account Summary", "Positions", "Account History"
    ])
    
    # Get and store account data
    if st.button("Refresh Account Data", key="refresh_account_button_unique"):
        with st.spinner("Fetching account data from Interactive Brokers..."):
            account_result = asyncio.run(fetch_ib_account_details())
            
            # If we got real data (not an error), store it in the database
            if account_result.get("success", False) and not account_result.get("sample_data", False):
                asyncio.run(store_account_history(account_result))
                st.success("Account data refreshed successfully!")
            else:
                if account_result.get("sample_data", False):
                    st.warning("Using sample data. Could not connect to Interactive Brokers.")
                    st.info(f"Error: {account_result.get('error', 'Unknown error')}")
                    # Use sample data for UI demonstration
                    account_result = get_sample_account_data()
                else:
                    st.error(f"Error fetching account data: {account_result.get('error', 'Unknown error')}")
        
            # Store in session state for tab access
            st.session_state.account_data = account_result
    
    # Check if we have account data in session state
    if 'account_data' not in st.session_state:
        st.info("Click 'Refresh Account Data' to load account information.")
        account_result = get_sample_account_data()  # Use sample data initially
        st.session_state.account_data = account_result
    
    # Account Summary Tab
    with acc_tab1:
        st.subheader("Account Summary")
        
        # Add refresh button
        refresh_account = st.button("Refresh Account Data", key="refresh_account_in_modal")
        
        # Get account details
        try:
            account_details = asyncio.run(fetch_ib_account_details())
            
            if account_details and 'summary' in account_details:
                # Create a DataFrame from account summary
                summary_data = account_details['summary']
                
                if summary_data:
                    # Convert to DataFrame
                    df_summary = pd.DataFrame(summary_data)
                    
                    # Display as metrics for important values
                    col1, col2, col3 = st.columns(3)
                    
                    # Net Liquidation Value
                    nlv = next((item['value'] for item in summary_data if item['tag'] == 'NetLiquidation'), 'N/A')
                    col1.metric("Net Liquidation Value", f"${nlv}")
                    
                    # Cash Balance
                    cash = next((item['value'] for item in summary_data if item['tag'] == 'TotalCashBalance'), 'N/A')
                    col2.metric("Cash Balance", f"${cash}")
                    
                    # Buying Power
                    bp = next((item['value'] for item in summary_data if item['tag'] == 'BuyingPower'), 'N/A')
                    col3.metric("Buying Power", f"${bp}")
                    
                    # Display full summary as a table
                    st.subheader("Complete Account Summary")
                    st.dataframe(
                        df_summary,
                        column_config={
                            "tag": "Metric",
                            "value": "Value", 
                            "currency": "Currency"
                        },
                        use_container_width=True
                    )
                else:
                    # If no real data, use sample data
                    st.warning("No live account data available - showing sample data")
                    sample_data = get_sample_account_data()
                    
                    # Display sample metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Net Liquidation Value", f"${sample_data['NetLiquidation']}")
                    col2.metric("Cash Balance", f"${sample_data['TotalCashBalance']}")
                    col3.metric("Buying Power", f"${sample_data['BuyingPower']}")
            else:
                st.warning("No account data available")
        except Exception as e:
            st.error(f"Error fetching account details: {e}")
            # Fall back to sample data
            st.warning("Using sample account data")
            sample_data = get_sample_account_data()
            
            # Display sample metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Net Liquidation Value", f"${sample_data['NetLiquidation']}")
            col2.metric("Cash Balance", f"${sample_data['TotalCashBalance']}")
            col3.metric("Buying Power", f"${sample_data['BuyingPower']}")
    
    # Positions Tab
    with acc_tab2:
        st.subheader("Current Positions")
        
        if 'account_data' in st.session_state:
            positions = st.session_state.account_data.get("positions", [])
            
            if positions:
                # Convert to DataFrame
                pos_df = pd.DataFrame(positions)
                
                # Calculate current value (will be estimated if using sample data)
                pos_df['currentPrice'] = pos_df.apply(
                    lambda row: row['avgCost'] * (1 + (0.05 * np.random.randn())), axis=1
                )
                pos_df['marketValue'] = pos_df['position'] * pos_df['currentPrice']
                pos_df['costBasis'] = pos_df['position'] * pos_df['avgCost']
                pos_df['unrealizedPnL'] = pos_df['marketValue'] - pos_df['costBasis']
                pos_df['unrealizedPnLPct'] = (pos_df['unrealizedPnL'] / pos_df['costBasis']) * 100
                
                # Position type (long/short)
                pos_df['positionType'] = pos_df['position'].apply(
                    lambda x: "LONG" if x > 0 else "SHORT" if x < 0 else "NONE"
                )
                
                # Display positions
                st.dataframe(
                    pos_df[[
                        'symbol', 'exchange', 'position', 'positionType', 'avgCost', 
                        'currentPrice', 'marketValue', 'unrealizedPnL', 'unrealizedPnLPct'
                    ]].style.format({
                        'avgCost': '${:.2f}',
                        'currentPrice': '${:.2f}',
                        'marketValue': '${:.2f}',
                        'unrealizedPnL': '${:.2f}',
                        'unrealizedPnLPct': '{:.2f}%'
                    }).applymap(
                        lambda val: 'color: green' if val > 0 else 'color: red' if val < 0 else '',
                        subset=['unrealizedPnL', 'unrealizedPnLPct']
                    ),
                    use_container_width=True
                )
                
                # Create visualization of positions
                st.subheader("Position Visualization")
                
                # Market value by symbol
                fig = go.Figure()
                
                for i, row in pos_df.iterrows():
                    color = "green" if row['positionType'] == "LONG" else "red"
                    
                    fig.add_trace(go.Bar(
                        x=[row['symbol']],
                        y=[abs(row['marketValue'])],
                        name=f"{row['symbol']} ({row['positionType']})",
                        marker_color=color,
                        text=f"${abs(row['marketValue']):,.2f}",
                        textposition="auto"
                    ))
                
                fig.update_layout(
                    title="Position Size by Symbol",
                    xaxis_title="Symbol",
                    yaxis_title="Market Value ($)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # P&L by position
                pos_df_sorted = pos_df.sort_values('unrealizedPnL', ascending=False)
                
                # Create a bar chart colored by P&L
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=pos_df_sorted['symbol'],
                    y=pos_df_sorted['unrealizedPnL'],
                    marker_color=pos_df_sorted['unrealizedPnL'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    ),
                    text=pos_df_sorted['unrealizedPnL'].apply(lambda x: f"${x:,.2f}"),
                    textposition="auto"
                ))
                
                fig.update_layout(
                    title="Unrealized P&L by Position",
                    xaxis_title="Symbol",
                    yaxis_title="Unrealized P&L ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pie chart of portfolio allocation
                fig = go.Figure(data=[go.Pie(
                    labels=pos_df['symbol'],
                    values=pos_df['marketValue'].abs(),
                    textinfo='label+percent',
                    hoverinfo='label+value+percent'
                )])
                
                fig.update_layout(
                    title="Portfolio Allocation by Symbol"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No position data available. Please refresh.")
        else:
            st.info("No position data available. Please refresh.")
    
    # Account History Tab
    with acc_tab3:
        st.subheader("Account History")
        
        # Get duration for history view
        history_days = st.slider("Number of days to display", 1, 90, 30)
        
        if st.button("Fetch Account History"):
            with st.spinner("Fetching account history..."):
                history = asyncio.run(fetch_account_history(history_days))
                
                if history:
                    # Convert to DataFrame
                    history_df = pd.DataFrame([dict(r) for r in history])
                    
                    # Display history
                    st.dataframe(
                        history_df[[
                            'timestamp', 'account_id', 'net_liquidation', 'total_cash',
                            'buying_power', 'unrealized_pnl', 'realized_pnl'
                        ]].style.format({
                            'net_liquidation': '${:,.2f}',
                            'total_cash': '${:,.2f}',
                            'buying_power': '${:,.2f}',
                            'unrealized_pnl': '${:,.2f}',
                            'realized_pnl': '${:,.2f}',
                            'timestamp': '{:%Y-%m-%d %H:%M:%S}'
                        }),
                        use_container_width=True
                    )
                    
                    # Create account value chart
                    st.subheader("Account Value History")
                    
                    # Line chart of account value over time
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['net_liquidation'],
                        mode='lines+markers',
                        name='Net Liquidation Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add total cash as a second line
                    fig.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['total_cash'],
                        mode='lines',
                        name='Total Cash',
                        line=dict(color='green', width=1.5, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Account Value History',
                        xaxis_title='Date',
                        yaxis_title='Value ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create P&L history chart
                    st.subheader("P&L History")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['unrealized_pnl'],
                        mode='lines',
                        name='Unrealized P&L',
                        line=dict(color='orange', width=1.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['realized_pnl'],
                        mode='lines',
                        name='Realized P&L',
                        line=dict(color='blue', width=1.5)
                    ))
                    
                    # Add a zero line for reference
                    fig.add_shape(
                        type="line",
                        xref="paper",
                        yref="y",
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=0,
                        line=dict(color="gray", width=1, dash="dash")
                    )
                    
                    fig.update_layout(
                        title='Profit & Loss History',
                        xaxis_title='Date',
                        yaxis_title='P&L ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display daily changes
                    st.subheader("Daily Account Changes")
                    
                    # Calculate daily changes
                    history_df['date'] = history_df['timestamp'].dt.date
                    
                    # Group by date and get first and last value for each day
                    daily_df = history_df.groupby('date').agg({
                        'net_liquidation': ['first', 'last'],
                        'unrealized_pnl': ['first', 'last'],
                        'realized_pnl': ['first', 'last']
                    }).reset_index()
                    
                    # Flatten multi-level columns
                    daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
                    
                    # Calculate changes
                    daily_df['net_change'] = daily_df['net_liquidation_last'] - daily_df['net_liquidation_first']
                    daily_df['net_change_pct'] = (daily_df['net_change'] / daily_df['net_liquidation_first']) * 100
                    daily_df['unrealized_change'] = daily_df['unrealized_pnl_last'] - daily_df['unrealized_pnl_first']
                    daily_df['realized_change'] = daily_df['realized_pnl_last'] - daily_df['realized_pnl_first']
                    
                    # Display daily changes
                    st.dataframe(
                        daily_df[[
                            'date', 'net_liquidation_first', 'net_liquidation_last', 
                            'net_change', 'net_change_pct', 'unrealized_change', 'realized_change'
                        ]].style.format({
                            'net_liquidation_first': '${:,.2f}',
                            'net_liquidation_last': '${:,.2f}',
                            'net_change': '${:,.2f}',
                            'net_change_pct': '{:,.2f}%',
                            'unrealized_change': '${:,.2f}',
                            'realized_change': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Bar chart of daily changes
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=daily_df['date'],
                        y=daily_df['net_change'],
                        marker_color=daily_df['net_change'].apply(
                            lambda x: 'green' if x > 0 else 'red'
                        ),
                        text=daily_df['net_change'].apply(lambda x: f"${x:,.2f}"),
                        textposition="outside"
                    ))
                    
                    fig.update_layout(
                        title='Daily Account Value Changes',
                        xaxis_title='Date',
                        yaxis_title='Change ($)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No account history available. Please make sure the account_history table exists and contains data.")
                    
                    # Create sample data for demo purposes
                    st.warning("Displaying sample data for demonstration purposes.")
                    
                    # Generate sample data
                    dates = pd.date_range(end=datetime.now(), periods=history_days)
                    sample_history = []
                    
                    base_value = 50000
                    current_value = base_value
                    
                    for date in dates:
                        # Random daily change -2% to +2%
                        daily_change = current_value * (np.random.random() * 0.04 - 0.02)
                        current_value += daily_change
                        
                        # Add some noise for intraday fluctuations
                        for hour in range(0, 24, 6):
                            if hour > 0:
                                intraday_change = current_value * (np.random.random() * 0.01 - 0.005)
                                current_value += intraday_change
                            
                            sample_history.append({
                                'timestamp': date.replace(hour=hour),
                                'account_id': 'SAMPLE',
                                'net_liquidation': current_value,
                                'total_cash': current_value * 0.4,
                                'buying_power': current_value * 2,
                                'unrealized_pnl': current_value * 0.05 * (np.random.random() * 2 - 1),
                                'realized_pnl': current_value * 0.02 * (np.random.random() * 2 - 1)
                            })
                    
                    # Convert to DataFrame
                    sample_df = pd.DataFrame(sample_history)
                    
                    # Display sample history
                    st.dataframe(
                        sample_df[[
                            'timestamp', 'account_id', 'net_liquidation', 'total_cash',
                            'buying_power', 'unrealized_pnl', 'realized_pnl'
                        ]].style.format({
                            'net_liquidation': '${:,.2f}',
                            'total_cash': '${:,.2f}',
                            'buying_power': '${:,.2f}',
                            'unrealized_pnl': '${:,.2f}',
                            'realized_pnl': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Create sample account value chart
                    st.subheader("Sample Account Value History")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=sample_df['timestamp'],
                        y=sample_df['net_liquidation'],
                        mode='lines',
                        name='Net Liquidation Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=sample_df['timestamp'],
                        y=sample_df['total_cash'],
                        mode='lines',
                        name='Total Cash',
                        line=dict(color='green', width=1.5, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Sample Account Value History',
                        xaxis_title='Date',
                        yaxis_title='Value ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def fund_allocation_dashboard():
    """
    Display a real-time dashboard for the fund allocation system, showing:
    - Active bots and their rankings
    - Current trades
    - Fund allocation
    - Manual trade control
    """
    st.title("Fund Allocation System Dashboard")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Active Bots", "Current Trades", "Manual Controls"])
    
    # Define internal function to get bot rankings with required columns
    async def dashboard_fetch_bot_rankings():
        """Fetch bot rankings with guaranteed columns for the dashboard"""
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # Check if table exists
                table_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'bot_rankings'
                    );
                """)
                
                if not table_exists:
                    # Return empty dataframe with required columns
                    return pd.DataFrame(columns=['bot_id', 'rank_score', 'is_active'])
                
                # Check if rank_score column exists
                rank_score_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'bot_rankings' AND column_name = 'rank_score'
                    );
                """)
                
                # Check if is_active column exists
                is_active_exists = await pool.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'bot_rankings' AND column_name = 'is_active'
                    );
                """)
                
                # Build query dynamically based on available columns
                select_columns = ["bot_id"]
                if rank_score_exists:
                    select_columns.append("rank_score")
                if is_active_exists:
                    select_columns.append("is_active")
                
                query = f"""
                    SELECT {', '.join(select_columns)}
                    FROM bot_rankings
                    ORDER BY bot_id
                """
                
                # Execute query
                rankings = await pool.fetch(query)
                
                # Convert to DataFrame
                df = pd.DataFrame(rankings)
                
                # Add missing columns with default values
                if 'rank_score' not in df.columns and len(df) > 0:
                    df['rank_score'] = 5.0  # Default rank score
                if 'is_active' not in df.columns and len(df) > 0:
                    df['is_active'] = True  # Default to active
                
                return df
        except Exception as e:
            st.error(f"Error fetching bot rankings: {e}")
            # Return empty dataframe with required columns
            return pd.DataFrame(columns=['bot_id', 'rank_score', 'is_active'])
    
    with tab1:
        st.header("Bot Rankings & Allocation")
        
        # Create two columns for bot metrics and allocation
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Bot Rankings Table
            st.subheader("Bot Rankings")
            
            # Add refresh button
            refresh_rankings = st.button("Refresh Rankings", key="refresh_rankings")
            
            # Run async function to get bot rankings data using the dashboard-specific function
            bot_rankings = asyncio.run(dashboard_fetch_bot_rankings())
            
            if bot_rankings is not None and len(bot_rankings) > 0:
                # Format the rankings data for display
                df_rankings = bot_rankings
                
                # Add a visual indicator for active status
                def format_active(active):
                    if active:
                        return "✅ Active"
                    else:
                        return "❌ Inactive"
                
                # Check if 'is_active' column exists
                if 'is_active' in df_rankings.columns:
                    df_rankings['status'] = df_rankings['is_active'].apply(format_active)
                else:
                    # If 'is_active' doesn't exist, assume all bots are active
                    df_rankings['status'] = "✅ Active"
                
                # Format the score if column exists
                if 'rank_score' in df_rankings.columns:
                    df_rankings['rank_score'] = df_rankings['rank_score'].apply(lambda x: f"{x:.2f}")
                else:
                    # If 'rank_score' doesn't exist, create a placeholder
                    df_rankings['rank_score'] = "N/A"
                
                # Make sure we have required columns for display
                required_columns = ['bot_id', 'rank_score', 'status']
                missing_columns = [col for col in required_columns if col not in df_rankings.columns]
                
                # Add missing columns with placeholder values
                for col in missing_columns:
                    if col == 'bot_id':
                        # This should never happen, but add a safety check
                        df_rankings['bot_id'] = range(1, len(df_rankings) + 1)
                    else:
                        df_rankings[col] = "N/A"
                        
                # Now display the dataframe with known columns
                st.dataframe(
                    df_rankings[required_columns],
                    column_config={
                        "bot_id": "Bot ID",
                        "rank_score": "Rank Score",
                        "status": "Status"
                    },
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("No bot ranking data available")
        
        with col2:
            # Fund Allocation Pie Chart
            st.subheader("Fund Allocation")
            
            # Define a safer function for fund allocation that won't crash
            async def dashboard_fetch_fund_allocation(total_funds=10000):
                """Fetch fund allocation with error handling for the dashboard"""
                try:
                    # Get bot rankings with our safe function
                    df_rankings = await dashboard_fetch_bot_rankings()
                    
                    if df_rankings is None or len(df_rankings) == 0:
                        return []
                    
                    # Create fund allocation data
                    allocations = []
                    # Calculate total score of all active bots
                    is_active_filter = df_rankings['is_active'] if 'is_active' in df_rankings.columns else pd.Series([True] * len(df_rankings))
                    active_bots = df_rankings[is_active_filter]
                    
                    if len(active_bots) == 0:
                        return []
                        
                    # Get rank score or use default if missing
                    if 'rank_score' in active_bots.columns:
                        total_score = active_bots['rank_score'].sum()
                    else:
                        # If no rank scores, equal allocation
                        total_score = len(active_bots)
                        active_bots['rank_score'] = 1.0
                    
                    # Calculate allocations
                    for _, bot in active_bots.iterrows():
                        bot_id = bot['bot_id']
                        rank_score = bot['rank_score'] if 'rank_score' in bot else 1.0
                        
                        # Prevent division by zero
                        if total_score > 0:
                            percentage = (rank_score / total_score) * 100
                            allocated_amount = (rank_score / total_score) * total_funds
                        else:
                            # Equal allocation if all scores are zero
                            percentage = 100 / len(active_bots)
                            allocated_amount = total_funds / len(active_bots)
                        
                        allocations.append({
                            'bot_id': bot_id,
                            'percentage': percentage,
                            'allocated_amount': allocated_amount,
                            'rank_score': rank_score
                        })
                    
                    return allocations
                except Exception as e:
                    st.error(f"Error calculating fund allocation: {e}")
                    return []
            
            # Get fund allocation data with the safer function
            fund_allocation = asyncio.run(dashboard_fetch_fund_allocation())
            
            if fund_allocation and len(fund_allocation) > 0:
                # Create a DataFrame for the pie chart
                df_allocation = pd.DataFrame(fund_allocation)
                df_allocation = df_allocation.sort_values('percentage', ascending=False)
                
                # Create pie chart
                fig = px.pie(
                    df_allocation, 
                    values='percentage', 
                    names='bot_id', 
                    title='Fund Allocation by Bot',
                    hole=0.4
                )
                
                # Update trace information for better hovering details
                hover_template = (
                    "Bot ID: %{label}<br>" +
                    "Allocation: %{value:.1f}%<br>" +
                    "Amount: $%{customdata:,.2f}"
                )
                
                fig.update_traces(
                    hovertemplate=hover_template,
                    customdata=df_allocation['allocated_amount'],
                    textinfo='label+percent'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No fund allocation data available")
    
    with tab2:
        st.header("Current Open Trades")
        
        # Add refresh button
        refresh_trades = st.button("Refresh Trades", key="refresh_trades")
        
        # Safe function to get active trades
        async def get_active_trades_safely():
            try:
                db_pool = await create_db_pool()
                trade_manager = TradeManager(db_pool)
                active_trades = await trade_manager.get_active_trades()
                await db_pool.close()
                return active_trades
            except Exception as e:
                st.error(f"Error fetching active trades: {e}")
                return []
        
        # Get active trades
        active_trades = asyncio.run(get_active_trades_safely())
        
        if active_trades and len(active_trades) > 0:
            # Convert to DataFrame
            try:
                df_trades = pd.DataFrame(active_trades)
                
                # Format the trade direction with up/down arrows
                def format_direction(direction):
                    if direction == "LONG":
                        return "🔼 LONG"
                    else:
                        return "🔽 SHORT"
                
                if 'trade_direction' in df_trades.columns:
                    df_trades['direction'] = df_trades['trade_direction'].apply(format_direction)
                
                # Calculate time in trade
                def time_in_trade(entry_time):
                    if entry_time:
                        time_diff = datetime.now() - entry_time
                        hours = time_diff.total_seconds() / 3600
                        return f"{hours:.1f} hours"
                    return "Unknown"
                
                if 'entry_time' in df_trades.columns:
                    df_trades['time_in_trade'] = df_trades['entry_time'].apply(time_in_trade)
                
                # Select and reorder columns for display
                display_columns = ['trade_id', 'bot_id', 'ticker', 'direction', 
                                'entry_price', 'time_in_trade', 'rank_score']
                
                # Display only columns that exist
                existing_columns = [col for col in display_columns if col in df_trades.columns]
                
                # Ensure we have at least some columns to display
                if not existing_columns:
                    existing_columns = df_trades.columns.tolist()
                
                st.dataframe(
                    df_trades[existing_columns],
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error displaying trade data: {e}")
                st.info("Raw trade data:")
                st.write(active_trades)
        else:
            st.info("No active trades at the moment")
    
    with tab3:
        st.header("Manual Trade Controls")
        
        # Safe function to get active trades for the dropdown
        async def get_active_trades_for_dropdown():
            try:
                db_pool = await create_db_pool()
                trade_manager = TradeManager(db_pool)
                active_trades = await trade_manager.get_active_trades()
                await db_pool.close()
                return active_trades
            except Exception as e:
                st.error(f"Error fetching active trades for dropdown: {e}")
                return []
        
        # Get active trades for the dropdown
        active_trades = asyncio.run(get_active_trades_for_dropdown())
        
        if active_trades and len(active_trades) > 0:
            try:
                # Create a dropdown for trade selection
                trade_options = []
                for t in active_trades:
                    # Check if all required keys exist
                    bot_id = t.get('bot_id', 'Unknown')
                    ticker = t.get('ticker', 'Unknown')
                    direction = t.get('trade_direction', 'Unknown')
                    trade_id = t.get('trade_id', 0)
                    
                    # Create the option text
                    option_text = f"Trade {trade_id}: Bot {bot_id} - {ticker} {direction}"
                    trade_options.append(option_text)
                
                if not trade_options:
                    st.warning("No trade details available")
                    return
                
                selected_trade = st.selectbox(
                    "Select trade to close:",
                    options=trade_options,
                    index=0
                )
                
                # Extract trade_id from the selected option
                try:
                    selected_trade_id = int(selected_trade.split(":")[0].replace("Trade ", ""))
                except (ValueError, IndexError, AttributeError):
                    st.error("Could not parse trade ID from selection")
                    selected_trade_id = None
                
                if selected_trade_id is not None:
                    # Current market price input
                    current_price = st.number_input(
                        "Current Market Price (for manual close)",
                        min_value=0.01,
                        step=0.01,
                        value=100.00
                    )
                    
                    # Close trade button
                    if st.button("Close Selected Trade", key="close_trade_btn"):
                        with st.spinner("Closing trade..."):
                            try:
                                # Execute the trade close using a coroutine wrapper
                                async def close_trade_coroutine():
                                    db_pool = await create_db_pool()
                                    trade_manager = TradeManager(db_pool)
                                    result = await trade_manager.complete_trade(selected_trade_id, current_price)
                                    await db_pool.close()
                                    return result
                                
                                # Run the coroutine
                                result = asyncio.run(close_trade_coroutine())
                                
                                if result and result.get('success', False):
                                    st.success(f"Successfully closed trade {selected_trade_id}")
                                    
                                    # Show trade details
                                    if 'pnl' in result:
                                        pnl = result['pnl']
                                        if pnl > 0:
                                            st.metric("Trade Result", f"${pnl:.2f}", delta=f"{pnl:.2f}")
                                        else:
                                            st.metric("Trade Result", f"${pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                                else:
                                    st.error(f"Failed to close trade: {result.get('reason', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error closing trade: {e}")
            except Exception as e:
                st.error(f"Error setting up trade controls: {e}")
                st.info("Raw trade data:")
                st.write(active_trades)
        else:
            st.warning("No active trades available to close")
        
        # Add a section for closing all trades
        st.subheader("Emergency Controls")
        
        if st.button("Close All Trades", key="close_all_trades"):
            # Show confirmation dialog
            confirm = st.text_input("Type 'CONFIRM' to close all trades:")
            
            if confirm.upper() == "CONFIRM":
                with st.spinner("Closing all trades..."):
                    # Execute close all trades
                    success = asyncio.run(emergency_close_all_trades())
                    if success:
                        st.success("Successfully closed all trades")
                    else:
                        st.error("Failed to close all trades")

async def emergency_close_all_trades():
    """Close all open trades"""
    try:
        db_pool = await create_db_pool()
        trade_manager = TradeManager(db_pool)
        
        # Get all active trades
        active_trades = await trade_manager.get_active_trades()
        
        success = True
        for trade in active_trades:
            # Use the current entry price as exit price (this could be improved)
            result = await trade_manager.complete_trade(trade['trade_id'], trade['entry_price'])
            if not result or not result.get('success', False):
                success = False
        
        await db_pool.close()
        return success
    except Exception as e:
        st.error(f"Error closing all trades: {e}")
        return False

def show_account_manager():
    """Display account information and allow managing account settings"""
    st.title("Account Manager")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Account Summary", "Positions", "Account History"])
    
    with tab1:
        st.header("Account Summary")
        
        # Add refresh button
        refresh_account = st.button("Refresh Account Data", key="account_summary_refresh")
        
        # Get account details
        try:
            account_details = asyncio.run(fetch_ib_account_details())
            
            if account_details and 'summary' in account_details:
                # Create a DataFrame from account summary
                summary_data = account_details['summary']
                
                if summary_data:
                    # Convert to DataFrame
                    df_summary = pd.DataFrame(summary_data)
                    
                    # Display as metrics for important values
                    col1, col2, col3 = st.columns(3)
                    
                    # Net Liquidation Value
                    nlv = next((item['value'] for item in summary_data if item['tag'] == 'NetLiquidation'), 'N/A')
                    col1.metric("Net Liquidation Value", f"${nlv}")
                    
                    # Cash Balance
                    cash = next((item['value'] for item in summary_data if item['tag'] == 'TotalCashBalance'), 'N/A')
                    col2.metric("Cash Balance", f"${cash}")
                    
                    # Buying Power
                    bp = next((item['value'] for item in summary_data if item['tag'] == 'BuyingPower'), 'N/A')
                    col3.metric("Buying Power", f"${bp}")
                    
                    # Display full summary as a table
                    st.subheader("Complete Account Summary")
                    st.dataframe(
                        df_summary,
                        column_config={
                            "tag": "Metric",
                            "value": "Value", 
                            "currency": "Currency"
                        },
                        use_container_width=True
                    )
                else:
                    # If no real data, use sample data
                    st.warning("No live account data available - showing sample data")
                    sample_data = get_sample_account_data()
                    
                    # Display sample metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Net Liquidation Value", f"${sample_data['NetLiquidation']}")
                    col2.metric("Cash Balance", f"${sample_data['TotalCashBalance']}")
                    col3.metric("Buying Power", f"${sample_data['BuyingPower']}")
            else:
                st.warning("No account data available")
        except Exception as e:
            st.error(f"Error fetching account details: {e}")
            # Fall back to sample data
            st.warning("Using sample account data")
            sample_data = get_sample_account_data()
            
            # Display sample metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Net Liquidation Value", f"${sample_data['NetLiquidation']}")
            col2.metric("Cash Balance", f"${sample_data['TotalCashBalance']}")
            col3.metric("Buying Power", f"${sample_data['BuyingPower']}")
    
    with tab2:
        st.header("Current Positions")
        
        try:
            account_details = asyncio.run(fetch_ib_account_details())
            
            if account_details and 'positions' in account_details and account_details['positions']:
                # Create DataFrame for positions
                positions = account_details['positions']
                df_positions = pd.DataFrame(positions)
                
                # Format and display the positions
                st.dataframe(
                    df_positions,
                    column_config={
                        "symbol": "Symbol",
                        "position": st.column_config.NumberColumn("Position", format="%.2f"),
                        "avgCost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
                        "marketValue": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                    },
                    use_container_width=True
                )
            else:
                st.info("No positions currently held")
        except Exception as e:
            st.error(f"Error fetching positions: {e}")
    
    with tab3:
        st.header("Account History")
        
        # Time period selection
        days = st.slider("History Period (Days)", min_value=7, max_value=90, value=30, step=1)
        
        try:
            # Fetch historical account data
            account_history = asyncio.run(fetch_account_history(days=days))
            
            if account_history and len(account_history) > 0:
                # Convert to DataFrame
                df_history = pd.DataFrame(account_history)
                
                # Convert timestamp to datetime
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                
                # Create account value chart
                st.subheader("Account Value History")
                
                fig = px.line(
                    df_history, 
                    x='timestamp', 
                    y='net_liquidation_value',
                    title=f'Account Value - Last {days} Days'
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Net Liquidation Value ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data in expandable section
                with st.expander("Show Raw Account History Data"):
                    st.dataframe(df_history)
            else:
                st.warning("No account history data available")
        except Exception as e:
            st.error(f"Error fetching account history: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    # Database check
    db_status = asyncio.run(check_database_schema())
    
    # Add the Fund Allocation Dashboard to navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
        
    # Create sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        st.write("### Main Views")
        
        if st.button("System Dashboard", key="nav_dash"):
            st.session_state.page = 'dashboard'
        
        if st.button("Bot Metrics", key="nav_metrics"):
            st.session_state.page = 'metrics'
        
        if st.button("Trade Analysis", key="nav_trade"):
            st.session_state.page = 'trade'
        
        if st.button("Variable Weights", key="nav_weights"):
            st.session_state.page = 'weights'
            
        if st.button("Fund Allocation Dashboard", key="nav_fund_alloc"):
            st.session_state.page = 'fund_allocation'
            
        if st.button("Account Manager", key="nav_account"):
            st.session_state.page = 'account'
        
        st.write("---")
        st.write("### Data Tools")
        
        if st.button("Raw Bot Metrics", key="nav_raw"):
            st.session_state.page = 'raw_metrics'
        
        if st.button("Raw Rankings", key="nav_rank"):
            st.session_state.page = 'raw_rankings'
        
        st.write("---")
        st.write("### System Controls")
        
        if st.button("Database Check", key="nav_db"):
            st.session_state.page = 'db_check'
    
    # Display the selected page based on session state
    if st.session_state.page == 'dashboard':
        show_dashboard()
    elif st.session_state.page == 'metrics':
        show_metrics()
    elif st.session_state.page == 'trade':
        trade_analysis()
    elif st.session_state.page == 'weights':
        weights_ui = WeightsManagementUI(DB_CONFIG)
        weights_ui.render()
    elif st.session_state.page == 'raw_metrics':
        show_raw_metrics()
    elif st.session_state.page == 'raw_rankings':
        show_raw_rankings()
    elif st.session_state.page == 'db_check':
        show_db_check()
    elif st.session_state.page == 'fund_allocation':
        fund_allocation_dashboard()
    elif st.session_state.page == 'account':
        show_account_manager()
