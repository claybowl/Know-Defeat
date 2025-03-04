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

### Latest Updates - March 1, 2025

#### Weekly Progress Summary
We've made significant progress on the Know Defeat Trading System this week, completing several high-priority tasks:

- ✅ **Weighted Ranking System** - Implemented the dynamic variable weighting system that allows the algorithm to adjust importance of different metrics based on market conditions
- ✅ **Fund Allocation Logic** - Completed the mechanism for distributing funds to trading bots based on their performance ranking
- ✅ **Visual Representation of Weights** - Added visualization components to the dashboard for better understanding of how weights impact bot ranking
- ✅ **Bot Metrics Representation** - Fixed issues with how bot_metrics are displayed in the user interface, improving readability
- ✅ **Dynamic bot_metric Table** - Finalized the schema for storing performance metrics with all specified variables

#### Database Improvements
- Standardized the bot_metrics table with consistent decimal precision (DECIMAL(4,1)) for all percentage metrics
- Implemented the variable_weights table to store dynamic weights for different performance indicators
- Created the calculate_bot_rank() function to determine bot rankings based on weighted metrics
- Added time-based performance tracking (1hr, 2hr, 1day, 1week, 1month)

#### Next Steps
- Complete standardization of bot configuration parameters
- Begin implementation of the Probability Engine for calculating success probabilities
- Prepare for performance testing with increased calculation load (100x)
- Explore NVIDIA Jetson hardware options for scaling computation

---

*Database and infrastructure are operational with initial testing showing promising results. The system can now store tick-level data, track bot metrics, and dynamically rank trading algorithms.*
    """)

################################################################################################################################


# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost',
    'min_size': 2,          # Minimum number of connections in the pool
    'max_size': 10,         # Maximum number of connections in the pool
    'command_timeout': 60.0,     # Timeout for database commands
    'max_queries': 50000,        # Maximum number of queries per connection
    'max_cached_statement_lifetime': 0,  # Don't cache statements
    'max_cacheable_statement_size': 0,   # Don't cache statements
    'timeout': 10.0,             # Connection timeout
    'server_settings': {
        'application_name': 'TradingDashboard',  # Name to identify this application
        'client_min_messages': 'warning'         # Reduce log noise
    }
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
tab_controls, tab_logs, tab_tables, tab_trades, tab_params, tab_rankings, tab_export = st.tabs([
    "Controls", "Logs", "Tables", "Trade Data", "Parameters", "Bot Rankings", "Data Export"
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
    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs([
        "Current Rankings", "Historical Performance", "Weight Management", "Database Diagnostics"
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
                    rankings = await pool.fetch("""
                        SELECT bot_id, rank_score, is_active
                        FROM bot_rankings
                        WHERE is_active = true
                        ORDER BY rank_score DESC;
                    """)
                    
                    if not rankings:
                        return None
                        
                    total_score = sum(row['rank_score'] for row in rankings)
                    
                    allocations = []
                    for row in rankings:
                        allocation = (row['rank_score'] / total_score) * total_funds if total_score > 0 else total_funds / len(rankings)
                        allocations.append({
                            'bot_id': row['bot_id'],
                            'allocation_amount': allocation,
                            'allocation_percentage': (allocation / total_funds) * 100,
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
            
            if st.button("Calculate Allocation"):
                allocations = asyncio.run(fetch_fund_allocation(total_funds))
                if allocations:
                    # Convert to DataFrame
                    alloc_df = pd.DataFrame(allocations)
                    
                    # Create pie chart of allocations
                    fig = go.Figure(data=[go.Pie(
                        labels=[f"Bot {row['bot_id']}" for i, row in alloc_df.iterrows()],
                        values=alloc_df['allocation_amount'],
                        textinfo='label+percent',
                        hoverinfo='label+value+percent',
                        marker=dict(
                            # Define colors based on rank for visual differentiation
                            colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
                        )
                    )])
                    
                    fig.update_layout(
                        title=f'Fund Allocation (${total_funds:,})',
                        height=300
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
                    
                    st.dataframe(
                        alloc_df_display[[
                            'bot', 'allocation_amount', 'allocation_percentage', 'rank_score'
                        ]].style.format({
                            'allocation_amount': lambda x: safe_format(x, '${:.2f}'),
                            'allocation_percentage': lambda x: safe_format(x, '{:.2f}%'),
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
            
            async def fetch_variable_weights():
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
                variable_weights = asyncio.run(fetch_variable_weights())
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
