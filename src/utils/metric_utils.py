import pandas as pd
from datetime import datetime, timedelta
from .db_utils import (
    execute_db_query,
    fetch_rows,
    create_db_pool,
    execute_query
)
import logging


def calculate_total_pnl(bot_id, algo_id):
    """
    Calculate total profit and loss for a trading bot strategy.
    
    Args:
        bot_id (int): Unique identifier for the trading bot
        algo_id (int): Algorithm version identifier
        
    Returns:
        float: Sum of all trade PNL values for the given bot/algorithm
    """
    query = """
        SELECT SUM(trade_pnl) AS total_pnl
        FROM sim_bot_trades
        WHERE bot_id = %s AND algo_id = %s;
    """
    total_pnl = execute_db_query(query, (bot_id, algo_id))
    return total_pnl

async def calculate_time_period_performance(db_pool, bot_id, ticker, hours=1):
    """
    Calculate normalized performance metric over a specified time window.
    
    Performance is calculated as average PNL per trade multiplied by 100,
    providing a percentage-based performance indicator.
    
    Args:
        db_pool (asyncpg.pool.Pool): Database connection pool
        bot_id (int): Trading bot identifier
        ticker (str): Financial instrument symbol (e.g., 'BTC-USD')
        hours (int): Time window duration in hours (default: 1)
        
    Returns:
        float: Normalized performance percentage
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow("""
            WITH period_trades AS (
                SELECT 
                    SUM(trade_pnl) as period_pnl,
                    COUNT(*) as trade_count
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND ticker = $2
                AND entry_time >= NOW() - interval '1 hour' * $3
                AND trade_status = 'closed'
            )
            SELECT 
                CASE 
                    WHEN trade_count > 0 THEN period_pnl / trade_count * 100
                    ELSE 0
                END as performance
            FROM period_trades
        """, bot_id, ticker, hours)
        
        return result['performance'] if result else 0.0

async def calculate_win_rate(db_pool, bot_id, ticker):
    """Calculate win rate for a bot"""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                WITH trade_stats AS (
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN trade_pnl > 0 THEN 1 END) as winning_trades
                    FROM sim_bot_trades
                    WHERE bot_id = $1 
                    AND ticker = $2
                    AND trade_status = 'closed'
                )
                SELECT 
                    CASE 
                        WHEN total_trades > 0 
                        THEN (winning_trades::float / total_trades::float) * 100
                        ELSE 0
                    END as win_rate
                FROM trade_stats
            """, bot_id, ticker)
            
            return result['win_rate'] if result else 0.0

    except Exception as e:
        print(f"Error calculating win rate: {e}")
        return 0.0

async def calculate_drawdown(db_pool, bot_id, ticker):
    """
    Calculate average capital loss during losing trades (drawdown).
    
    Drawdown is computed as the average absolute value of negative PNL trades,
    providing insight into typical loss magnitudes.
    
    Args:
        db_pool (asyncpg.pool.Pool): Database connection pool
        bot_id (int): Trading bot identifier
        ticker (str): Financial instrument symbol
        
    Returns:
        float: Average drawdown amount (absolute value)
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT AVG(ABS(trade_pnl)) as avg_drawdown
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND ticker = $2
                AND trade_pnl < 0
                AND trade_status = 'closed'
            """, bot_id, ticker)
            
            return abs(result['avg_drawdown']) if result and result['avg_drawdown'] else 0.0

    except Exception as e:
        print(f"Error calculating drawdown: {e}")
        return 0.0

async def calculate_profit_per_second(db_pool, bot_id, ticker):
    """
    Calculate profit generation efficiency metric.
    
    Measures how much profit is generated per second of trading activity,
    useful for comparing strategy efficiency across different timeframes.
    
    Args:
        db_pool (asyncpg.pool.Pool): Database connection pool
        bot_id (int): Trading bot identifier
        ticker (str): Financial instrument symbol
        
    Returns:
        float: Profit per second ratio (total PNL / total trade duration)
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow("""
            WITH trade_durations AS (
                SELECT 
                    trade_pnl,
                    EXTRACT(EPOCH FROM (exit_time - entry_time)) as duration_seconds
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND ticker = $2
                AND trade_status = 'closed'
                AND entry_time >= NOW() - interval '1 hour'
            )
            SELECT 
                COALESCE(SUM(trade_pnl) / NULLIF(SUM(duration_seconds), 0), 0) as profit_per_second
            FROM trade_durations
        """, bot_id, ticker)
        
        return result['profit_per_second'] if result else 0.0

async def calculate_win_streaks(db_pool, bot_id, ticker):
    """
    Analyze consecutive winning trade sequences.
    
    Tracks occurrences of different win streak lengths (2-5 consecutive wins)
    in the most recent 100 trades. Helps identify momentum patterns.
    
    Returns:
        dict: Counts of streaks meeting different length thresholds
    """
    try:
        async with db_pool.acquire() as conn:
            # Get the trades ordered by time
            trades = await conn.fetch("""
                SELECT 
                    trade_pnl > 0 as is_win,
                    trade_pnl
                FROM sim_bot_trades 
                WHERE bot_id = $1 
                AND ticker = $2
                AND trade_status = 'closed'
                ORDER BY entry_time DESC
                LIMIT 100
            """, bot_id, ticker)
            
            # Initialize counters
            current_streak = 0
            streaks = {
                'streak_2': 0,
                'streak_3': 0,
                'streak_4': 0,
                'streak_5': 0
            }
            
            # Count consecutive wins
            for trade in trades:
                if trade['is_win']:
                    current_streak += 1
                else:
                    # Update streak counters
                    if current_streak >= 2:
                        streaks['streak_2'] += 1
                    if current_streak >= 3:
                        streaks['streak_3'] += 1
                    if current_streak >= 4:
                        streaks['streak_4'] += 1
                    if current_streak >= 5:
                        streaks['streak_5'] += 1
                    current_streak = 0
            
            # Don't forget to count the last streak if it's still ongoing
            if current_streak >= 2:
                streaks['streak_2'] += 1
            if current_streak >= 3:
                streaks['streak_3'] += 1
            if current_streak >= 4:
                streaks['streak_4'] += 1
            if current_streak >= 5:
                streaks['streak_5'] += 1
                
            return streaks

    except Exception as e:
        print(f"Error calculating win streaks: {e}")
        return {
            'streak_2': 0,
            'streak_3': 0,
            'streak_4': 0,
            'streak_5': 0
        }

async def update_bot_metrics(db_pool, bot_id, ticker):
    """
    Main metrics aggregation and persistence function.
    
    Orchestrates calculation of all key performance indicators and stores
    them in the bot_metrics table. Implements upsert logic to maintain
    a single record per bot with latest metrics.
    
    Metrics collected:
    - Time-based performance (1h, 2h, 1d, 1w, 1m)
    - Win rate percentage
    - Average drawdown
    - Profit generation efficiency
    - Win streak probabilities
    - Total accumulated PNL
    
    Args:
        db_pool (asyncpg.pool.Pool): Database connection pool
        bot_id (int): Trading bot identifier
        ticker (str): Financial instrument symbol
    """
    try:
        # Calculate existing metrics
        one_hour = await calculate_time_period_performance(db_pool, bot_id, ticker, 1)
        two_hour = await calculate_time_period_performance(db_pool, bot_id, ticker, 2)
        one_day = await calculate_time_period_performance(db_pool, bot_id, ticker, 24)
        one_week = await calculate_time_period_performance(db_pool, bot_id, ticker, 24 * 7)
        one_month = await calculate_time_period_performance(db_pool, bot_id, ticker, 24 * 30)
        
        win_rate = await calculate_win_rate(db_pool, bot_id, ticker)
        drawdown = await calculate_drawdown(db_pool, bot_id, ticker)
        profit_per_sec = await calculate_profit_per_second(db_pool, bot_id, ticker)
        win_streaks = await calculate_win_streaks(db_pool, bot_id, ticker)
        
        # Calculate total PNL
        async with db_pool.acquire() as conn:
            total_pnl_result = await conn.fetchrow("""
                SELECT COALESCE(SUM(trade_pnl), 0) as total_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND ticker = $2
                AND trade_status = 'closed'
            """, bot_id, ticker)
            total_pnl = total_pnl_result['total_pnl']

        # Insert with all columns
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO bot_metrics (
                    bot_id, ticker, timestamp,
                    one_hour_performance, two_hour_performance,
                    one_day_performance, one_week_performance,
                    one_month_performance, win_rate,
                    avg_drawdown, total_pnl,
                    two_win_streak_prob, three_win_streak_prob,
                    four_win_streak_prob, last_updated
                ) VALUES ($1, $2, CURRENT_TIMESTAMP, 
                    $3, $4, $5, $6, $7, $8, $9, $10, 
                    $11, $12, $13, CURRENT_TIMESTAMP)
                ON CONFLICT (bot_id) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    timestamp = CURRENT_TIMESTAMP,
                    one_hour_performance = EXCLUDED.one_hour_performance,
                    two_hour_performance = EXCLUDED.two_hour_performance,
                    one_day_performance = EXCLUDED.one_day_performance,
                    one_week_performance = EXCLUDED.one_week_performance,
                    one_month_performance = EXCLUDED.one_month_performance,
                    win_rate = EXCLUDED.win_rate,
                    avg_drawdown = EXCLUDED.avg_drawdown,
                    total_pnl = EXCLUDED.total_pnl,
                    two_win_streak_prob = EXCLUDED.two_win_streak_prob,
                    three_win_streak_prob = EXCLUDED.three_win_streak_prob,
                    four_win_streak_prob = EXCLUDED.four_win_streak_prob,
                    last_updated = CURRENT_TIMESTAMP;
            """, bot_id, ticker, 
                one_hour, two_hour, one_day, one_week, one_month,
                win_rate, drawdown, total_pnl,
                win_streaks['streak_2'], win_streaks['streak_3'],
                win_streaks['streak_4'])

    except Exception as e:
        print(f"Error updating metrics for bot {bot_id}: {str(e)}")
        raise
