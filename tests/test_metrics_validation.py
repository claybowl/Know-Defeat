"""
Metrics Validation Test

This script creates predictable test data with known outcomes,
then compares direct SQL calculations against the metrics system
to validate accuracy of all metrics.
"""

import asyncio
import asyncpg
import logging
import sys
import os
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from tabulate import tabulate
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.metrics_calculator import MetricsCalculator
from src.metrics_updater import MetricsUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def setup_test_data(conn, test_bot_id=9999, test_ticker="TEST"):
    """Generate predictable test data for metrics validation"""
    logger.info(f"Setting up test data for bot ID {test_bot_id} and ticker {test_ticker}")
    
    # First, remove any existing test data to ensure clean state
    await conn.execute(f"""
        DELETE FROM sim_bot_trades WHERE bot_id = {test_bot_id};
        DELETE FROM sim_bots WHERE bot_id = {test_bot_id};
        DELETE FROM bot_metrics WHERE bot_id = {test_bot_id};
    """)
    
    # Create test bot
    await conn.execute(f"""
        INSERT INTO sim_bots (bot_id, ticker, algorithm_type, is_active, name, algorithm_module, 
                             trade_direction, position_size, trailing_stop_pct, description, 
                             version, created_at, last_updated)
        VALUES ({test_bot_id}, '{test_ticker}', 'TEST_ALGORITHM', true, 'Test Bot', 'test_module',
               'BOTH', 1000, 0.5, 'Test bot for metrics validation', '1.0', 
               NOW(), NOW())
        ON CONFLICT (bot_id) DO NOTHING;
    """)
    
    # Define patterns for predictable trades
    # Pattern 1: Alternating win/loss with fixed sizes and predictable patterns
    base_timestamp = datetime.now() - timedelta(days=60)
    trade_id_start = 50000  # Starting trade ID for test trades
    
    # Generate trades for different time periods to test period-based metrics
    trades = []
    
    # Basic pattern: alternating win/loss with increasing PnL values
    # Use periods that ensure we have trades in each time bucket (1hr, 2hr, 1day, 1week, 1month)
    
    # Trades from 2 months ago (for monthly metrics)
    for i in range(10):
        entry_time = base_timestamp + timedelta(days=1*i)
        exit_time = entry_time + timedelta(minutes=30)
        # Alternating win/loss with known profit/loss
        pnl = 10.0 if i % 2 == 0 else -5.0
        entry_price = 100.0
        exit_price = entry_price + pnl
        
        trades.append({
            'trade_id': trade_id_start + i,
            'bot_id': test_bot_id,
            'ticker': test_ticker,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_pnl': pnl,
            'trade_size': 1.0,
            'trade_direction': 'LONG',
            'trade_status': 'closed'
        })
    
    # Trades from 1-4 weeks ago (for weekly metrics)
    for i in range(20):
        entry_time = base_timestamp + timedelta(days=30) + timedelta(days=i*0.5)
        exit_time = entry_time + timedelta(minutes=45)
        # Create a pattern with 3 wins followed by 1 loss
        pnl = 15.0 if i % 4 != 3 else -10.0
        entry_price = 110.0
        exit_price = entry_price + pnl
        
        trades.append({
            'trade_id': trade_id_start + 10 + i,
            'bot_id': test_bot_id,
            'ticker': test_ticker,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_pnl': pnl,
            'trade_size': 1.0,
            'trade_direction': 'LONG',
            'trade_status': 'closed'
        })
    
    # Trades from last few days (for daily metrics)
    for i in range(24):
        entry_time = datetime.now() - timedelta(days=3) + timedelta(hours=i)
        exit_time = entry_time + timedelta(minutes=20)
        # Series of 5 wins, then 3 losses
        pnl = 20.0 if i % 8 < 5 else -15.0
        entry_price = 120.0
        exit_price = entry_price + pnl
        
        trades.append({
            'trade_id': trade_id_start + 30 + i,
            'bot_id': test_bot_id,
            'ticker': test_ticker,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_pnl': pnl,
            'trade_size': 1.0,
            'trade_direction': 'LONG',
            'trade_status': 'closed'
        })
    
    # Trades from last few hours (for hourly metrics)
    for i in range(12):
        entry_time = datetime.now() - timedelta(hours=3) + timedelta(minutes=i*15)
        exit_time = entry_time + timedelta(minutes=5)
        # All wins in the recent period to create a trend
        pnl = 25.0
        entry_price = 130.0
        exit_price = entry_price + pnl
        
        trades.append({
            'trade_id': trade_id_start + 54 + i,
            'bot_id': test_bot_id,
            'ticker': test_ticker,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_pnl': pnl,
            'trade_size': 1.0,
            'trade_direction': 'LONG',
            'trade_status': 'closed'
        })
    
    # Add a currently open trade
    entry_time = datetime.now() - timedelta(minutes=10)
    trades.append({
        'trade_id': trade_id_start + 66,
        'bot_id': test_bot_id,
        'ticker': test_ticker,
        'entry_time': entry_time,
        'exit_time': None,
        'entry_price': 140.0,
        'exit_price': None,
        'trade_pnl': None,
        'trade_size': 1.0,
        'trade_direction': 'LONG',
        'trade_status': 'open'
    })
    
    # Insert the test trades
    insert_query = """
        INSERT INTO sim_bot_trades (
            trade_id, bot_id, ticker, entry_time, exit_time, 
            entry_price, exit_price, trade_pnl, trade_size, 
            trade_direction, trade_status, algo_id,
            entry_trigger_price, exit_trigger_price,
            entry_trigger_time, exit_trigger_time,
            pnl_percent
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13, $14, $15, $16, $17
        );
    """
    
    for trade in trades:
        # Calculate pnl_percent for completed trades
        pnl_percent = None
        if trade['trade_pnl'] is not None and trade['entry_price'] is not None and trade['entry_price'] != 0:
            pnl_percent = (trade['trade_pnl'] / (trade['entry_price'] * trade['trade_size'])) * 100
            
        await conn.execute(
            insert_query,
            trade['trade_id'], trade['bot_id'], trade['ticker'],
            trade['entry_time'], trade['exit_time'],
            trade['entry_price'], trade['exit_price'], trade['trade_pnl'],
            trade['trade_size'], trade['trade_direction'], trade['trade_status'],
            trade['bot_id'],  # Using bot_id as algo_id
            None,  # entry_trigger_price
            None,  # exit_trigger_price
            None,  # entry_trigger_time
            None,  # exit_trigger_time
            pnl_percent
        )
    
    logger.info(f"Successfully inserted {len(trades)} test trades for bot {test_bot_id}")
    
    # Return the expected metric values based on our test data
    expected_metrics = calculate_expected_metrics(trades)
    return expected_metrics

def calculate_expected_metrics(trades):
    """Calculate the expected metric values manually from our test data"""
    # Filter closed trades only
    closed_trades = [t for t in trades if t['trade_status'] == 'closed']
    
    # Calculate basic metrics
    total_trades = len(closed_trades)
    winning_trades = len([t for t in closed_trades if t['trade_pnl'] > 0])
    total_pnl = sum(t['trade_pnl'] for t in closed_trades)
    
    # Calculate win rate
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Calculate avg_profit_per_trade
    avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
    
    # Filter trades for different time periods
    now = datetime.now()
    one_hour_trades = [t for t in closed_trades if t['entry_time'] >= now - timedelta(hours=1)]
    two_hour_trades = [t for t in closed_trades if t['entry_time'] >= now - timedelta(hours=2)]
    one_day_trades = [t for t in closed_trades if t['entry_time'] >= now - timedelta(days=1)]
    one_week_trades = [t for t in closed_trades if t['entry_time'] >= now - timedelta(weeks=1)]
    one_month_trades = [t for t in closed_trades if t['entry_time'] >= now - timedelta(days=30)]
    
    # Calculate period performances
    one_hour_perf = sum(t['trade_pnl'] for t in one_hour_trades)
    two_hour_perf = sum(t['trade_pnl'] for t in two_hour_trades)
    one_day_perf = sum(t['trade_pnl'] for t in one_day_trades)
    one_week_perf = sum(t['trade_pnl'] for t in one_week_trades)
    one_month_perf = sum(t['trade_pnl'] for t in one_month_trades)
    
    # Calculate profit factor
    winning_pnl = sum(t['trade_pnl'] for t in closed_trades if t['trade_pnl'] > 0)
    losing_pnl = abs(sum(t['trade_pnl'] for t in closed_trades if t['trade_pnl'] < 0))
    profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 1.0
    
    # Calculate drawdowns
    running_pnl = 0
    peak_pnl = 0
    drawdowns = []
    
    for trade in sorted(closed_trades, key=lambda x: x['entry_time']):
        running_pnl += trade['trade_pnl']
        
        if running_pnl > peak_pnl:
            peak_pnl = running_pnl
        else:
            drawdown = peak_pnl - running_pnl
            drawdowns.append(drawdown)
    
    avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0
    max_drawdown = max(drawdowns) if drawdowns else 0
    
    # Win streaks
    current_streak = 0
    streaks = {2: 0, 3: 0, 4: 0, 5: 0}
    
    for trade in sorted(closed_trades, key=lambda x: x['entry_time']):
        if trade['trade_pnl'] > 0:
            current_streak += 1
            
            # Check if this extends a streak of interest
            for streak_len in [2, 3, 4, 5]:
                if current_streak >= streak_len:
                    streaks[streak_len] += 1
        else:
            current_streak = 0
    
    # Calculate win streak percentages
    win_streak_2 = (streaks[2] / total_trades) * 100 if total_trades > 0 else 0
    win_streak_3 = (streaks[3] / total_trades) * 100 if total_trades > 0 else 0
    win_streak_4 = (streaks[4] / total_trades) * 100 if total_trades > 0 else 0
    win_streak_5 = (streaks[5] / total_trades) * 100 if total_trades > 0 else 0
    
    # Calculate average trade duration
    durations_seconds = [(t['exit_time'] - t['entry_time']).total_seconds() for t in closed_trades]
    avg_trade_duration = sum(durations_seconds) / len(durations_seconds) if durations_seconds else 0
    
    # Calculate profit per second
    first_trade_time = min([t['entry_time'] for t in closed_trades])
    last_trade_time = max([t['exit_time'] for t in closed_trades])
    trading_period_seconds = (last_trade_time - first_trade_time).total_seconds()
    profit_per_second = total_pnl / trading_period_seconds if trading_period_seconds > 0 else 0
    
    # Simplified model scores (based on your implementation)
    # These are less precise since your implementations use other metrics
    price_model_score = 50 + (win_rate * 0.3)
    volume_model_score = 50 + (win_rate * 0.2)
    price_wall_score = 50 + (profit_factor * 2)
    
    # Sharpe ratio calculation
    # Group trades by day
    trades_by_day = {}
    for trade in closed_trades:
        day = trade['entry_time'].date()
        if day not in trades_by_day:
            trades_by_day[day] = []
        trades_by_day[day].append(trade)
    
    # Calculate daily returns
    daily_returns = [sum(t['trade_pnl'] for t in day_trades) for day_trades in trades_by_day.values()]
    
    # Calculate Sharpe ratio
    avg_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    risk_free_rate = 0.02  # Same as in your implementation
    std_dev = np.std(daily_returns) if len(daily_returns) > 1 else 1
    sharpe_ratio = ((avg_return - risk_free_rate) / std_dev) if std_dev > 0 else 0
    
    # Return all expected metric values
    return {
        'total_trades': total_trades,
        'total_pnl': round(total_pnl, 2),
        'avg_win_rate': round(win_rate, 2),
        'one_hour_performance': round(one_hour_perf, 2),
        'two_hour_performance': round(two_hour_perf, 2),
        'one_day_performance': round(one_day_perf, 2),
        'one_week_performance': round(one_week_perf, 2),
        'one_month_performance': round(one_month_perf, 2),
        'avg_profit_per_trade': round(avg_profit_per_trade, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_drawdown': round(avg_drawdown, 2),
        'max_drawdown': round(max_drawdown, 2),
        'win_streak_2': round(win_streak_2, 2),
        'win_streak_3': round(win_streak_3, 2),
        'win_streak_4': round(win_streak_4, 2),
        'win_streak_5': round(win_streak_5, 2),
        'avg_trade_duration': round(avg_trade_duration, 2),
        'profit_per_second': round(profit_per_second, 4),
        'price_model_score': round(price_model_score, 2),
        'volume_model_score': round(volume_model_score, 2),
        'price_wall_score': round(price_wall_score, 2),
        'sharpe_ratio': round(sharpe_ratio, 4)
    }

async def run_direct_sql_calculations(conn, test_bot_id, test_ticker):
    """Calculate metrics using direct SQL queries as a reference"""
    logger.info(f"Running direct SQL calculations for bot {test_bot_id}")
    
    # Total trades
    total_trades = await conn.fetchval(f"""
        SELECT COUNT(*) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    # Total PnL
    total_pnl = await conn.fetchval(f"""
        SELECT SUM(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    # Win rate
    win_rate = await conn.fetchval(f"""
        SELECT 
            (COUNT(CASE WHEN trade_pnl > 0 THEN 1 END) * 100.0 / COUNT(*)) 
        FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    # Period performances
    one_hour_perf = await conn.fetchval(f"""
        SELECT SUM(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
        AND entry_time >= NOW() - INTERVAL '1 hour'
    """)
    
    two_hour_perf = await conn.fetchval(f"""
        SELECT SUM(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
        AND entry_time >= NOW() - INTERVAL '2 hour'
    """)
    
    one_day_perf = await conn.fetchval(f"""
        SELECT SUM(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
        AND entry_time >= NOW() - INTERVAL '1 day'
    """)
    
    one_week_perf = await conn.fetchval(f"""
        SELECT SUM(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
        AND entry_time >= NOW() - INTERVAL '1 week'
    """)
    
    one_month_perf = await conn.fetchval(f"""
        SELECT SUM(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
        AND entry_time >= NOW() - INTERVAL '30 days'
    """)
    
    # Average profit per trade
    avg_profit_per_trade = await conn.fetchval(f"""
        SELECT AVG(trade_pnl) FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    # Profit factor
    profit_factor = await conn.fetchval(f"""
        SELECT 
            SUM(CASE WHEN trade_pnl > 0 THEN trade_pnl ELSE 0 END) /
            NULLIF(ABS(SUM(CASE WHEN trade_pnl < 0 THEN trade_pnl ELSE 0 END)), 0) 
        FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    # Average trade duration
    avg_trade_duration = await conn.fetchval(f"""
        SELECT 
            EXTRACT(EPOCH FROM AVG(exit_time - entry_time))
        FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    # Profit per second
    profit_per_second = await conn.fetchval(f"""
        SELECT 
            SUM(trade_pnl) / 
            NULLIF(EXTRACT(EPOCH FROM (MAX(exit_time) - MIN(entry_time))), 0)
        FROM sim_bot_trades 
        WHERE bot_id = {test_bot_id} AND trade_status = 'closed'
    """)
    
    sql_results = {
        'total_trades': total_trades,
        'total_pnl': round(float(total_pnl) if total_pnl is not None else 0, 2),
        'avg_win_rate': round(float(win_rate) if win_rate is not None else 0, 2),
        'one_hour_performance': round(float(one_hour_perf) if one_hour_perf is not None else 0, 2),
        'two_hour_performance': round(float(two_hour_perf) if two_hour_perf is not None else 0, 2),
        'one_day_performance': round(float(one_day_perf) if one_day_perf is not None else 0, 2),
        'one_week_performance': round(float(one_week_perf) if one_week_perf is not None else 0, 2),
        'one_month_performance': round(float(one_month_perf) if one_month_perf is not None else 0, 2),
        'avg_profit_per_trade': round(float(avg_profit_per_trade) if avg_profit_per_trade is not None else 0, 2),
        'profit_factor': round(float(profit_factor) if profit_factor is not None else 0, 2),
        'avg_trade_duration': round(float(avg_trade_duration) if avg_trade_duration is not None else 0, 2),
        'profit_per_second': round(float(profit_per_second) if profit_per_second is not None else 0, 4)
    }
    
    return sql_results

async def run_metrics_calculator(pool, test_bot_id, test_ticker):
    """Run the metrics calculator on test data"""
    logger.info(f"Running metrics calculator for bot {test_bot_id}")
    
    # Create metrics calculator instance
    metrics_calculator = MetricsCalculator(pool)
    metrics_updater = MetricsUpdater(pool, metrics_calculator)
    
    # Run all metrics calculations
    result = await metrics_updater.update_bot_metrics(test_bot_id, test_ticker)
    
    if not result:
        logger.error("Metrics updater failed to update metrics")
        return {}
    
    # Fetch the calculated metrics from database
    async with pool.acquire() as conn:
        metrics_row = await conn.fetchrow(f"""
            SELECT * FROM bot_metrics 
            WHERE bot_id = {test_bot_id}
            ORDER BY timestamp DESC
            LIMIT 1
        """)
    
    if not metrics_row:
        logger.error("No metrics found in database after calculation")
        return {}
    
    # Convert to dictionary and round numeric values
    metrics_dict = {}
    for key in metrics_row.keys():
        if key in ('bot_id', 'ticker', 'algo_id', 'timestamp', 'last_updated'):
            continue  # Skip non-metric fields
            
        value = metrics_row[key]
        
        # Handle None values
        if value is None:
            metrics_dict[key] = 0
            continue
            
        # Handle numeric values
        if isinstance(value, (int, float, Decimal)):
            if key in ('profit_per_second', 'sharpe_ratio', 'average_true_range'):
                metrics_dict[key] = round(float(value), 4)
            else:
                metrics_dict[key] = round(float(value), 2)
        else:
            # For interval types, convert to seconds
            if isinstance(value, timedelta):
                metrics_dict[key] = round(value.total_seconds(), 2)
            else:
                metrics_dict[key] = value
    
    return metrics_dict

def compare_metrics(expected, sql_direct, calculator):
    """Compare the three sets of metric calculations and report discrepancies"""
    logger.info("Comparing metric calculations")
    
    comparison_data = []
    
    # Determine which metrics to compare based on what's available
    all_metrics = set(expected.keys()) | set(sql_direct.keys()) | set(calculator.keys())
    
    for metric in sorted(all_metrics):
        # Get values or use "N/A" if not present
        expected_value = expected.get(metric, "N/A")
        sql_value = sql_direct.get(metric, "N/A")
        calc_value = calculator.get(metric, "N/A")
        
        # Calculate discrepancies
        if metric in expected and metric in calculator:
            if isinstance(expected_value, (int, float)) and isinstance(calc_value, (int, float)):
                exp_val = float(expected_value)
                calc_val = float(calc_value)
                
                abs_diff = abs(exp_val - calc_val)
                rel_diff = abs_diff / max(abs(exp_val), 0.0001) * 100 if exp_val != 0 else (0 if calc_val == 0 else 100)
                
                status = "✅" if rel_diff < 5 else "⚠️" if rel_diff < 15 else "❌"
                diff_str = f"{abs_diff:.2f} ({rel_diff:.1f}%)"
            else:
                status = "❓"
                diff_str = "N/A"
        else:
            status = "❓"
            diff_str = "N/A"
        
        comparison_data.append({
            'Metric': metric,
            'Expected': expected_value,
            'SQL Direct': sql_value,
            'Calculator': calc_value,
            'Difference': diff_str,
            'Status': status
        })
    
    # Create results DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate summary stats
    total_metrics = len(comparison_data)
    passed = sum(1 for row in comparison_data if row['Status'] == "✅")
    warning = sum(1 for row in comparison_data if row['Status'] == "⚠️")
    failed = sum(1 for row in comparison_data if row['Status'] == "❌")
    unknown = sum(1 for row in comparison_data if row['Status'] == "❓")
    
    success_rate = (passed / total_metrics) * 100 if total_metrics > 0 else 0
    
    return {
        'comparison_df': comparison_df,
        'total_metrics': total_metrics,
        'passed': passed,
        'warning': warning,
        'failed': failed,
        'unknown': unknown,
        'success_rate': success_rate
    }

async def run_test():
    """Run the metrics validation test"""
    print("=== Metrics Validation Test ===")
    print("This test creates predictable data with known outcomes and validates metric calculations")
    
    # Create pool
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        print("\n1. Setting up test data...")
        async with pool.acquire() as conn:
            # Set up test data
            expected_metrics = await setup_test_data(conn)
            
            # Run direct SQL calculations
            print("\n2. Running direct SQL calculations...")
            sql_results = await run_direct_sql_calculations(conn, 9999, "TEST")
            
            # Run metrics calculator
            print("\n3. Running metrics calculator...")
            calculator_results = await run_metrics_calculator(pool, 9999, "TEST")
            
            # Compare results
            print("\n4. Comparing results...")
            comparison = compare_metrics(expected_metrics, sql_results, calculator_results)
            
            # Print comparison table
            print("\n=== Metrics Comparison ===")
            print(tabulate.tabulate(comparison['comparison_df'], headers='keys', tablefmt='psql', showindex=False))
            
            # Print summary
            print("\n=== Validation Summary ===")
            print(f"Total metrics tested:  {comparison['total_metrics']}")
            print(f"✅ Passed:             {comparison['passed']} ({comparison['passed']/comparison['total_metrics']*100:.1f}%)")
            print(f"⚠️ Warnings:           {comparison['warning']} ({comparison['warning']/comparison['total_metrics']*100:.1f}%)")
            print(f"❌ Failed:             {comparison['failed']} ({comparison['failed']/comparison['total_metrics']*100:.1f}%)")
            print(f"❓ Unknown:            {comparison['unknown']} ({comparison['unknown']/comparison['total_metrics']*100:.1f}%)")
            print(f"Overall success rate:  {comparison['success_rate']:.1f}%")
            
            # Save detailed results
            comparison['comparison_df'].to_csv("metrics_validation_results.csv", index=False)
            print("\nDetailed results saved to metrics_validation_results.csv")
            
            # Cleanup test data
            print("\n5. Cleaning up test data...")
            await conn.execute(f"""
                DELETE FROM sim_bot_trades WHERE bot_id = 9999;
                DELETE FROM sim_bots WHERE bot_id = 9999;
                DELETE FROM bot_metrics WHERE bot_id = 9999;
            """)
    
    except Exception as e:
        logger.error(f"Error during metrics validation test: {e}")
        print(f"\n❌ Error: {e}")
    
    finally:
        # Close pool
        await pool.close()
        print("\nTest completed.")

if __name__ == "__main__":
    # Verify dependencies
    missing_deps = []
    try:
        import asyncpg
    except ImportError:
        missing_deps.append("asyncpg")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import tabulate
    except ImportError:
        missing_deps.append("tabulate")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("❌ Missing dependencies. Please install:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        sys.exit(1)
    
    # Run the test
    asyncio.run(run_test())