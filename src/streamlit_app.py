import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import subprocess
import asyncio
import asyncpg
import sys
import os
from datetime import datetime, timedelta
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Set page config
st.set_page_config(page_title="Trading System Dashboard", layout="wide")

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.ib_controller_process = None
    # Update to include 8 bots
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

st.title("Trading System Dashboard")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

def start_ib_controller():
    """Start the IB Controller process"""
    try:
        if not st.session_state.ib_controller_process:
            process = subprocess.Popen(
                ['python', os.path.join('src', 'ib_controller.py')],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
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
            script_path = os.path.join('src', 'bots', f'{bot_name}_bot.py')
            # If the script filename differs (e.g., ..._bot2.py), you might need logic to pick the right filename
            if not os.path.exists(script_path):  
                # Fallback for "2" version naming
                script_path_2 = os.path.join('src', 'bots', f'{bot_name}.py')
                if os.path.exists(script_path_2):
                    script_path = script_path_2

            process = subprocess.Popen(
                ['python', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
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
        else:
            process = st.session_state.ib_controller_process
            if process:
                process.terminate()
                process.wait(timeout=5)
                st.session_state.ib_controller_process = None
                st.session_state.log_buffer['ib_controller'] = []
    except Exception as e:
        st.error(f"Error stopping process: {e}")

def update_logs():
    """Update log buffers for all running processes"""
    try:
        # Update IB Controller logs
        if st.session_state.ib_controller_process:
            while True:
                line = st.session_state.ib_controller_process.stdout.readline()
                if not line:
                    break
                st.session_state.log_buffer['ib_controller'].append(line.strip())

        # Update bot logs
        for bot_name, process in st.session_state.bot_processes.items():
            if process:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    st.session_state.log_buffer[bot_name].append(line.strip())
    except Exception as e:
        st.error(f"Error updating logs: {e}")

# Create main sections using tabs
tab_controls, tab_logs, tab_tables, tab_trades, tab_params, tab_export = st.tabs([
    "Controls", "Logs", "Tables", "Trade Data", "Parameters", "Data Export"
])

# Controls Section
with tab_controls:
    st.header("System Controls")

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
            for bot_name in st.session_state.bot_processes.keys():
                if not start_bot(bot_name):
                    success = False
            if success:
                st.success("All bots started successfully!")

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

    # Display IB Controller logs
    st.subheader("IB Controller Logs")
    if st.session_state.log_buffer['ib_controller']:
        st.text_area(
            "IB Controller Output",
            value="\n".join(st.session_state.log_buffer['ib_controller'][-100:]),
            height=200
        )

    # Display bot logs
    st.subheader("Bot Logs")
    for bot_name in st.session_state.bot_processes.keys():
        if st.session_state.log_buffer[bot_name]:
            st.text_area(
                f"{bot_name} Bot Output",
                value="\n".join(st.session_state.log_buffer[bot_name][-100:]),
                height=200
            )

# Tables Section
with tab_tables:
    st.header("Database Tables")

    async def fetch_data():
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            # Fetch trade data
            trades = await pool.fetch("""
                SELECT * FROM sim_bot_trades 
                ORDER BY trade_timestamp DESC 
                LIMIT 100
            """)

            # Fetch tick data
            ticks = await pool.fetch("""
                SELECT * FROM tick_data 
                ORDER BY timestamp DESC 
                LIMIT 100
            """)

            return trades, ticks

    if st.button("Refresh Data"):
        try:
            trades, ticks = asyncio.run(fetch_data())
            # Display trade data
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df)

            # Display tick data
            st.subheader("Recent Ticks")
            ticks_df = pd.DataFrame(ticks)
            st.dataframe(ticks_df)

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Trade Data Section
with tab_trades:
    st.header("Trade Analysis")

    async def fetch_trade_stats():
        async with asyncpg.create_pool(**DB_CONFIG) as pool:
            stats = await pool.fetch("""
                SELECT 
                    symbol,
                    trade_direction,
                    COUNT(*) as total_trades,
                    AVG(profit_loss) as avg_pnl,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades
                FROM sim_bot_trades
                GROUP BY symbol, trade_direction
            """)
            return stats

    if st.button("Calculate Statistics"):
        try:
            stats = asyncio.run(fetch_trade_stats())
            stats_df = pd.DataFrame(stats)

            # Display statistics
            st.subheader("Trading Statistics")
            st.dataframe(stats_df)

            # Create visualization
            if not stats_df.empty:
                fig = go.Figure()
                for symbol in stats_df['symbol'].unique():
                    symbol_data = stats_df[stats_df['symbol'] == symbol]
                    fig.add_trace(go.Bar(
                        name=symbol,
                        x=symbol_data['trade_direction'],
                        y=symbol_data['avg_pnl']
                    ))

                fig.update_layout(
                    title="Average PnL by Symbol and Direction",
                    xaxis_title="Trade Direction",
                    yaxis_title="Average PnL"
                )

                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error calculating statistics: {e}")

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

    with col_export2:
        if st.button("Export Tick Data to CSV"):
            try:
                # Run the export_tick_data.py script
                subprocess.Popen(['python', 'export_tick_data.py'])
                st.success("Exporting tick data... Check the console or logs for status.")
            except Exception as e:
                st.error(f"Error running export_tick_data.py: {e}") 