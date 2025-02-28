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
import psycopg2
import logging
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from weights_management_ui import WeightsManagementUI

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
                    
                    # Display detailed statistics
                    st.subheader("Detailed Statistics by Bot")
                    st.dataframe(
                        stats_df.style.format({
                            'trade_count': '{:,}',
                            'winning_trades': '{:,}',
                            'losing_trades': '{:,}',
                            'avg_pnl': '${:.2f}',
                            'total_pnl': '${:.2f}',
                            'calculated_win_rate': '{:.1f}%',
                            'avg_duration': '{:.1f}'
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
                                    metrics_df[col] = metrics_df[col].map('{:.2%}'.format)
                                else:
                                    metrics_df[col] = metrics_df[col].map('{:.2f}'.format)
                        
                        st.dataframe(metrics_df)
                        
                        # Create performance heatmap for available performance metrics
                        performance_cols = [col for col in metrics_df.columns if 'performance' in col]
                        if performance_cols:
                            heatmap_data = metrics_df[['bot_id', 'ticker'] + performance_cols].copy()
                            for col in performance_cols:
                                heatmap_data[col] = pd.to_numeric(heatmap_data[col].str.rstrip('%'), errors='coerce')
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=heatmap_data[performance_cols].values,
                                x=performance_cols,
                                y=heatmap_data.apply(lambda x: f"Bot {x['bot_id']} - {x['ticker']}", axis=1),
                                colorscale='RdYlGn'
                            ))
                            fig.update_layout(title='Performance Heatmap Across Timeframes')
                            st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("Quick Stats")
                if 'metrics_df' in locals():
                    try:
                        # Calculate and display key statistics for available metrics
                        if 'one_day_performance' in metrics_df.columns:
                            best_performer = metrics_df.loc[pd.to_numeric(metrics_df['one_day_performance'].str.rstrip('%'), errors='coerce').idxmax()]
                            st.metric("Best 24h Performer", 
                                    f"Bot {best_performer['bot_id']} - {best_performer['ticker']}", 
                                    best_performer['one_day_performance'])
                        
                        if 'avg_win_rate' in metrics_df.columns:
                            highest_win_rate = metrics_df.loc[pd.to_numeric(metrics_df['avg_win_rate'].str.rstrip('%'), errors='coerce').idxmax()]
                            st.metric("Highest Win Rate", 
                                    f"Bot {highest_win_rate['bot_id']} - {highest_win_rate['ticker']}", 
                                    highest_win_rate['avg_win_rate'])
                        
                        if 'total_pnl' in metrics_df.columns:
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
                    
                    for ticker in metrics_df['ticker'].unique():
                        ticker_data = metrics_df[metrics_df['ticker'] == ticker]
                        
                        # Convert string percentages to float
                        profit_factor = pd.to_numeric(ticker_data['profit_factor'], errors='coerce')
                        drawdown = pd.to_numeric(ticker_data['max_drawdown'].str.rstrip('%'), errors='coerce')
                        
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
                    
                    # Win Streak Analysis
                    streak_cols = ['win_streak_2', 'win_streak_3', 'win_streak_4', 'win_streak_5']
                    streak_data = metrics_df[['bot_id', 'ticker'] + streak_cols].copy()
                    
                    for col in streak_cols:
                        streak_data[col] = pd.to_numeric(streak_data[col].str.rstrip('%'), errors='coerce')
                    
                    fig = go.Figure()
                    for idx, row in streak_data.iterrows():
                        fig.add_trace(go.Bar(
                            name=f"Bot {row['bot_id']} - {row['ticker']}",
                            x=['2 Wins', '3 Wins', '4 Wins'],
                            y=[row[col] for col in streak_cols]
                        ))
                    
                    fig.update_layout(
                        title='Win Streak Probability Analysis',
                        barmode='group',
                        xaxis_title='Streak Length',
                        yaxis_title='Probability (%)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

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
                        hour_perf = pd.to_numeric(metrics_df['one_hour_performance'].str.rstrip('%'), errors='coerce').mean()
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
                    
                    with col2:
                        # Win Rate Gauge
                        win_rate = pd.to_numeric(metrics_df['avg_win_rate'].str.rstrip('%'), errors='coerce').mean()
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = win_rate,
                            title = {'text': "Win Rate"},
                            gauge = {'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkgreen"}}))
                        st.plotly_chart(fig)
                    
                    with col3:
                        # Profit Factor Gauge
                        profit_factor = pd.to_numeric(metrics_df['profit_factor'], errors='coerce').mean()
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = profit_factor,
                            title = {'text': "Profit Factor"},
                            gauge = {'axis': {'range': [0, 3]},
                                    'bar': {'color': "darkorange"}}))
                        st.plotly_chart(fig)
                    
                    # Add real-time trade frequency chart
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
                                    variable_name VARCHAR(50) NOT NULL,
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
                        text=weights_df['weight'].apply(lambda x: f'{x:.1f}%'),
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title='Variable Weights Distribution',
                        xaxis_title='Weight (%)',
                        yaxis_title='Variable Name',
                        height=600,
                        showlegend=False,
                        xaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display weights table
                    st.dataframe(
                        weights_df.style.format({
                            'weight': '{:.1f}%',
                            'last_updated': '{:%Y-%m-%d %H:%M:%S}'
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
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_weights.get(variable_name, 5.0),
                step=0.1,
                help="Enter the weight percentage for this variable (0-100)"
            )
            
            # Update button
            if st.button("Update Weight"):
                try:
                    async def update_variable_weight():
                        async with asyncpg.create_pool(**DB_CONFIG) as pool:
                            await pool.execute("""
                                INSERT INTO variable_weights (variable_name, weight, last_updated)
                                VALUES ($1, $2, NOW())
                                ON CONFLICT (variable_name) 
                                DO UPDATE SET 
                                    weight = $2,
                                    last_updated = NOW();
                            """, variable_name, weight)
                    
                    asyncio.run(update_variable_weight())
                    st.success(f"Weight for {variable_name} updated to {weight:.1f}%")
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
                        return total or 0
                except Exception as e:
                    st.error(f"Error calculating total weight: {str(e)}")
                    return 0
            
            total_weight = asyncio.run(get_total_weight())
            st.metric(
                "Total Weight",
                f"{total_weight:.1f}%",
                delta=f"{total_weight - 100:.1f}%" if total_weight != 100 else None,
                delta_color="inverse"
            )
            
            if abs(total_weight - 100) > 0.1:
                st.warning("⚠️ Total weight should sum to 100%")
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
                '1hr Perf': '{:.2f}%',
                '24hr Perf': '{:.2f}%',
                'Win Rate': '{:.1f}%',
                'Profit/sec': '${:.4f}'
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
                'Avg PNL': '${:.2f}',
                'Avg Duration (sec)': '{:.1f}'
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

        metrics = asyncio.run(fetch_bot_metrics())
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(
                metrics_df.style.format({
                    'one_hour_performance': '{:.2f}%',
                    'one_day_performance': '{:.2f}%',
                    'avg_win_rate': '{:.1f}%',
                    'profit_per_second': '${:.4f}'
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
                'Avg PNL': '${:.2f}',
                'Avg Duration (sec)': '{:.1f}'
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
