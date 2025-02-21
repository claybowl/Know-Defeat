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
from plotly.subplots import make_subplots


# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Set page config
st.set_page_config(page_title="Trading System Dashboard", layout="wide")

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
                # Extract column names from the first row
                column_names = [k for k in trades[0].keys()]
                trades_df = pd.DataFrame(trades, columns=column_names)
                
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

    async def fetch_trade_stats():
        try:
            async with asyncpg.create_pool(**DB_CONFIG) as pool:
                # Get bot_metrics table structure
                columns = await pool.fetch("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'bot_metrics'
                    ORDER BY ordinal_position;
                """)

                # Get trade statistics
                stats = await pool.fetch("""
                    SELECT 
                        bot_id,
                        ticker as symbol,
                        trade_direction,
                        COUNT(*) as trade_count,
                        ROUND(AVG(trade_size)::numeric, 2) as avg_trade_size,
                        ROUND(AVG(entry_price)::numeric, 2) as avg_entry_price,
                        ROUND(AVG(exit_price)::numeric, 2) as avg_exit_price,
                        ROUND(AVG(trade_pnl)::numeric, 2) as avg_pnl,
                        ROUND(SUM(trade_pnl)::numeric, 2) as total_pnl,
                        ROUND(COUNT(*) FILTER (WHERE trade_pnl > 0)::numeric * 100.0 / 
                            NULLIF(COUNT(*), 0), 2) as win_rate
                    FROM sim_bot_trades
                    WHERE trade_status = 'closed'
                      AND exit_price IS NOT NULL
                      AND entry_price IS NOT NULL
                      AND trade_size IS NOT NULL
                    GROUP BY bot_id, ticker, trade_direction
                    ORDER BY bot_id ASC, ticker ASC, trade_direction ASC;
                """)
                
                # Also fetch the latest bot metrics
                metrics = await pool.fetch("""
                    SELECT 
                        bot_id,
                        ticker,
                        algo_id,
                        one_hour_performance,
                        two_hour_performance,
                        one_day_performance,
                        one_week_performance,
                        one_month_performance,
                        win_rate,
                        avg_drawdown,
                        max_drawdown,
                        profit_factor,
                        avg_profit_per_trade,
                        total_pnl,
                        price_model_score,
                        volume_model_score,
                        price_wall_score,
                        two_win_streak_prob,
                        three_win_streak_prob,
                        four_win_streak_prob,
                        avg_trade_duration,
                        trade_frequency,
                        avg_trade_size,
                        market_participation_rate
                    FROM bot_metrics
                    ORDER BY total_pnl DESC;
                """)
                
                if not stats:
                    raise Exception("No statistics could be calculated from the trades")
                    
                return stats, columns, metrics

        except Exception as e:
            st.error(f"Database error: {str(e)}")
            print(f"Full error: {str(e)}")
            return None, None, None

    if st.button("Calculate Statistics", key="calc_stats_button"):
        try:
            stats, columns, metrics = asyncio.run(fetch_trade_stats())
            
            if stats and len(stats) > 0:
                # Create DataFrame with explicit column names first
                stats_df = pd.DataFrame(stats, columns=[
                    'bot_id', 'symbol', 'trade_direction', 'trade_count',
                    'avg_trade_size', 'avg_entry_price', 'avg_exit_price',
                    'avg_pnl', 'total_pnl', 'win_rate'
                ])
                
                # Display raw data for debugging
                st.subheader("Raw Trade Statistics")
                st.write("Number of records:", stats_df.shape[0])
                
                # Create three columns for the metadata display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Columns:")
                    columns_df = pd.DataFrame({
                        'Index': range(len(stats_df.columns)),
                        'Column Name': stats_df.columns
                    })
                    st.dataframe(columns_df, hide_index=True)
                
                with col2:
                    st.write("Bot Metrics:")
                    if columns:
                        columns_df = pd.DataFrame(columns, columns=['Column Name', 'Data Type', 'Nullable'])
                        st.dataframe(columns_df)
                    else:
                        st.info("No bot metrics structure available")
                
                with col3:
                    st.write("Data types:")
                    dtypes_df = pd.DataFrame({
                        'Column': stats_df.columns,
                        'Type': stats_df.dtypes.astype(str)
                    })
                    st.dataframe(dtypes_df, hide_index=True)
                
                # Convert numeric columns to float, replacing NULL/None with 0
                numeric_cols = ['trade_count', 'avg_trade_size', 'avg_entry_price', 
                              'avg_exit_price', 'avg_pnl', 'total_pnl', 'win_rate']
                for col in numeric_cols:
                    stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0)
                
                # Display statistics in a table
                st.subheader("Trading Statistics")
                st.dataframe(stats_df)

                if not stats_df.empty:
                    try:
                        # Create visualizations
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                "Average PnL by Symbol and Direction",
                                "Win Rate vs Average Trade Size",
                                "Trade Count by Symbol",
                                "PNL Distribution"
                            )
                        )
                        
                        # Plot 1: Bar chart for Average PnL
                        try:
                            symbols = stats_df['symbol'].unique()
                            for symbol in symbols:
                                data = stats_df[stats_df['symbol'] == symbol]
                                if not data.empty and 'trade_direction' in data.columns and 'avg_pnl' in data.columns:
                                    fig.add_trace(
                                        go.Bar(
                                            name=str(symbol),
                                            x=data['trade_direction'].astype(str),
                                            y=data['avg_pnl'].astype(float),
                                            text=data['avg_pnl'].round(2),
                                            textposition='auto',
                                        ),
                                        row=1, col=1
                                    )
                        except Exception as e:
                            st.warning(f"Could not create Average PnL chart: {str(e)}")
                        
                        # Plot 2: Scatter plot for Win Rate vs Trade Size
                        try:
                            for symbol in symbols:
                                data = stats_df[stats_df['symbol'] == symbol]
                                if not data.empty and all(col in data.columns for col in ['avg_trade_size', 'win_rate', 'trade_count']):
                                    fig.add_trace(
                                        go.Scatter(
                                            name=str(symbol),
                                            x=data['avg_trade_size'].astype(float),
                                            y=data['win_rate'].astype(float),
                                            mode='markers',
                                            marker=dict(
                                                size=data['trade_count'].astype(float) * 3,
                                                opacity=0.7
                                            ),
                                            text=data['trade_direction'],
                                            hovertemplate=(
                                                "Symbol: %{text}<br>" +
                                                "Avg Trade Size: $%{x:.2f}<br>" +
                                                "Win Rate: %{y:.1f}%"
                                            )
                                        ),
                                        row=1, col=2
                                    )
                        except Exception as e:
                            st.warning(f"Could not create Win Rate vs Trade Size chart: {str(e)}")
                        
                        # Plot 3: Bar chart for Trade Count
                        try:
                            if 'symbol' in stats_df.columns and 'trade_count' in stats_df.columns:
                                trade_counts = stats_df.groupby('symbol')['trade_count'].sum().reset_index()
                                fig.add_trace(
                                    go.Bar(
                                        name="Trade Count",
                                        x=trade_counts['symbol'].astype(str),
                                        y=trade_counts['trade_count'].astype(float),
                                        text=trade_counts['trade_count'],
                                        textposition='auto',
                                        marker_color='indianred'
                                    ),
                                    row=2, col=1
                                )
                        except Exception as e:
                            st.warning(f"Could not create Trade Count chart: {str(e)}")
                        
                        # Plot 4: Histogram for PnL Distribution
                        try:
                            if 'avg_pnl' in stats_df.columns:
                                pnl_data = stats_df['avg_pnl'].dropna().astype(float)
                                if not pnl_data.empty:
                                    fig.add_trace(
                                        go.Histogram(
                                            x=pnl_data,
                                            nbinsx=20,
                                            name="PNL Distribution",
                                            marker_color='lightslategray'
                                        ),
                                        row=2, col=2
                                    )
                        except Exception as e:
                            st.warning(f"Could not create PnL Distribution chart: {str(e)}")
                        
                        # Update layout with error handling
                        try:
                            fig.update_layout(
                                height=800,
                                title_text="Overall Trade Analysis",
                                showlegend=True,
                                barmode='group',
                                margin=dict(l=40, r=40, t=80, b=40)
                            )
                            
                            # Update axes labels
                            fig.update_xaxes(title_text="Trade Direction", row=1, col=1)
                            fig.update_yaxes(title_text="Average PnL ($)", row=1, col=1)
                            fig.update_xaxes(title_text="Average Trade Size ($)", row=1, col=2)
                            fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
                            fig.update_xaxes(title_text="Symbol", row=2, col=1)
                            fig.update_yaxes(title_text="Number of Trades", row=2, col=1)
                            fig.update_xaxes(title_text="PnL ($)", row=2, col=2)
                            fig.update_yaxes(title_text="Frequency", row=2, col=2)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error updating chart layout: {str(e)}")
                            
                    except Exception as viz_error:
                        st.error(f"Error in visualization creation: {str(viz_error)}")
                        # Display the raw data as a fallback
                        st.write("Raw Statistics Data:")
                        st.dataframe(stats_df)

                # Add Bot Metrics Display
                st.subheader("Bot Performance Metrics")
                if metrics and len(metrics) > 0:
                    metrics_df = pd.DataFrame(metrics)
                    
                    # Format percentages
                    pct_cols = ['one_hour_performance', 'two_hour_performance', 
                               'one_day_performance', 'one_week_performance', 
                               'one_month_performance', 'win_rate', 'avg_drawdown',
                               'max_drawdown', 'market_participation_rate',
                               'two_win_streak_prob', 'three_win_streak_prob', 
                               'four_win_streak_prob']
                    
                    # Format scores
                    score_cols = ['price_model_score', 'volume_model_score', 
                                 'price_wall_score', 'profit_factor']
                    
                    # Format money values
                    money_cols = ['total_pnl', 'avg_profit_per_trade', 'avg_trade_size']
                    
                    # Apply formatting
                    for col in pct_cols:
                        if col in metrics_df.columns:
                            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                    
                    for col in score_cols:
                        if col in metrics_df.columns:
                            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                    
                    for col in money_cols:
                        if col in metrics_df.columns:
                            metrics_df[col] = metrics_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                    
                    # Format time duration
                    if 'avg_trade_duration' in metrics_df.columns:
                        metrics_df['avg_trade_duration'] = metrics_df['avg_trade_duration'].apply(
                            lambda x: f"{x:.0f}s" if pd.notnull(x) else "N/A"
                        )
                    
                    st.dataframe(metrics_df)
            else:
                st.warning("No trade statistics available. Please check if there are closed trades in the database.")
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")
            st.error("Please check the database connection and make sure the sim_bot_trades table exists with trade data.")

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
