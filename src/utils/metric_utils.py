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
    query = """
        SELECT SUM(trade_pnl) AS total_pnl
        FROM sim_bot_trades
        WHERE bot_id = %s AND algo_id = %s;
    """
    total_pnl = execute_db_query(query, (bot_id, algo_id))
    return total_pnl

async def calculate_time_period_performance(db_pool, bot_id, ticker, hours=1):
    """Calculate performance over a specific time period"""
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
    """Calculate average win rate"""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT 
                COUNT(CASE WHEN trade_pnl > 0 THEN 1 END)::float / 
                NULLIF(COUNT(*), 0) * 100 as win_rate
            FROM sim_bot_trades
            WHERE bot_id = $1 
            AND ticker = $2
            AND trade_status = 'closed'
            AND entry_time >= NOW() - interval '24 hours'
        """, bot_id, ticker)
        
        return result['win_rate'] if result else 0.0

async def calculate_drawdown(db_pool, bot_id, ticker):
    """Calculate average drawdown"""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow("""
            WITH running_pnl AS (
                SELECT 
                    trade_pnl,
                    SUM(trade_pnl) OVER (ORDER BY entry_time) as cumulative_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND ticker = $2
                AND trade_status = 'closed'
                AND entry_time >= NOW() - interval '24 hours'
            )
            SELECT 
                COALESCE(ABS(MIN(cumulative_pnl) - MAX(cumulative_pnl)), 0) as max_drawdown
            FROM running_pnl
        """, bot_id, ticker)
        
        return result['max_drawdown'] if result else 0.0

async def calculate_profit_per_second(db_pool, bot_id, ticker):
    """Calculate profit per second"""
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
    """Calculate win streak percentages"""
    async with db_pool.acquire() as conn:
        # This query calculates streaks
        streaks = await conn.fetch("""
            WITH trades AS (
                SELECT 
                    trade_id,
                    CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END as win,
                    entry_time
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND ticker = $2
                AND trade_status = 'closed'
                ORDER BY entry_time
            ),
            streaks AS (
                SELECT 
                    trade_id,
                    win,
                    SUM(CASE WHEN win != LAG(win, 1, -1) OVER (ORDER BY entry_time) THEN 1 ELSE 0 END) 
                    OVER (ORDER BY entry_time) as streak_group
                FROM trades
            ),
            streak_lengths AS (
                SELECT 
                    streak_group,
                    COUNT(*) as streak_length
                FROM streaks
                WHERE win = 1  -- Only count winning streaks
                GROUP BY streak_group
            )
            SELECT 
                COUNT(CASE WHEN streak_length >= 2 THEN 1 END)::float / NULLIF(COUNT(*), 0) * 100 as streak_2,
                COUNT(CASE WHEN streak_length >= 3 THEN 1 END)::float / NULLIF(COUNT(*), 0) * 100 as streak_3,
                COUNT(CASE WHEN streak_length >= 4 THEN 1 END)::float / NULLIF(COUNT(*), 0) * 100 as streak_4,
                COUNT(CASE WHEN streak_length >= 5 THEN 1 END)::float / NULLIF(COUNT(*), 0) * 100 as streak_5
            FROM streak_lengths
        """, bot_id, ticker)
        
        return streaks[0] if streaks else {'streak_2': 0, 'streak_3': 0, 'streak_4': 0, 'streak_5': 0}

async def update_bot_metrics(db_pool, bot_id, ticker):
    """Update all metrics for a bot"""
    try:
        # Calculate all metrics
        one_hour = await calculate_time_period_performance(db_pool, bot_id, ticker, 1)
        two_hour = await calculate_time_period_performance(db_pool, bot_id, ticker, 2)
        one_day = await calculate_time_period_performance(db_pool, bot_id, ticker, 24)
        one_week = await calculate_time_period_performance(db_pool, bot_id, ticker, 24 * 7)
        one_month = await calculate_time_period_performance(db_pool, bot_id, ticker, 24 * 30)
        
        win_rate = await calculate_win_rate(db_pool, bot_id, ticker)
        drawdown = await calculate_drawdown(db_pool, bot_id, ticker)
        profit_per_sec = await calculate_profit_per_second(db_pool, bot_id, ticker)
        win_streaks = await calculate_win_streaks(db_pool, bot_id, ticker)

        # Update the metrics table
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO bot_metrics (
                    bot_id, ticker, timestamp,
                    one_hour_performance, two_hour_performance,
                    one_day_performance, one_week_performance,
                    one_month_performance, avg_win_rate,
                    avg_drawdown, profit_per_second,
                    win_streak_2, win_streak_3,
                    win_streak_4, win_streak_5,
                    last_updated
                ) VALUES ($1, $2, NOW(), $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW())
                ON CONFLICT (bot_id) DO UPDATE
                SET 
                    one_hour_performance = EXCLUDED.one_hour_performance,
                    two_hour_performance = EXCLUDED.two_hour_performance,
                    one_day_performance = EXCLUDED.one_day_performance,
                    one_week_performance = EXCLUDED.one_week_performance,
                    one_month_performance = EXCLUDED.one_month_performance,
                    avg_win_rate = EXCLUDED.avg_win_rate,
                    avg_drawdown = EXCLUDED.avg_drawdown,
                    profit_per_second = EXCLUDED.profit_per_second,
                    win_streak_2 = EXCLUDED.win_streak_2,
                    win_streak_3 = EXCLUDED.win_streak_3,
                    win_streak_4 = EXCLUDED.win_streak_4,
                    win_streak_5 = EXCLUDED.win_streak_5,
                    last_updated = NOW()
            """, bot_id, ticker, one_hour, two_hour, one_day, one_week, one_month, 
                win_rate, drawdown, profit_per_sec, 
                win_streaks['streak_2'], win_streaks['streak_3'], 
                win_streaks['streak_4'], win_streaks['streak_5'])

    except Exception as e:
        print(f"Error updating metrics for bot {bot_id}: {e}")
        raise
