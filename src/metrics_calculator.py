import asyncpg
import logging

class MetricsCalculator:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def calculate_one_hour_performance(self, bot_id, ticker):
        async with self.db_pool.acquire() as connection:
            result = await connection.fetchval("""
                SELECT AVG(performance)
                FROM trades
                WHERE bot_id = $1 AND ticker = $2
                AND timestamp >= NOW() - INTERVAL '1 hour';
            """, bot_id, ticker)
            return result or 0.0

    async def calculate_avg_win_rate(self, bot_id, ticker):
        async with self.db_pool.acquire() as connection:
            result = await connection.fetchval("""
                SELECT AVG(win_rate)
                FROM trades
                WHERE bot_id = $1 AND ticker = $2
                AND timestamp >= NOW() - INTERVAL '1 hour';
            """, bot_id, ticker)
            return result or 0.0

    async def calculate_total_pnl(db_pool, bot_id, algo_id):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT SUM(trade_pnl) AS total_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_avg_profit_per_trade(db_pool, bot_id, algo_id):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT AVG(trade_pnl) AS avg_profit_per_trade
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_total_trades(db_pool, bot_id, algo_id):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) AS total_trades
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_profit_factor(db_pool, bot_id, algo_id):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT SUM(CASE WHEN trade_pnl > 0 THEN trade_pnl ELSE 0 END) /
                    ABS(SUM(CASE WHEN trade_pnl < 0 THEN trade_pnl ELSE 0 END)) AS profit_factor
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_two_hour_performance(db_pool, bot_id, algo_id):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT 
                    (SUM(CASE WHEN timestamp >= NOW() - INTERVAL '2 hours' THEN trade_pnl ELSE 0 END) /
                    SUM(CASE WHEN timestamp >= NOW() - INTERVAL '2 hours' THEN 1 ELSE 0 END)) AS two_hour_performance
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result






# CREATE TABLE bot_metrics (
#     -- Identifiers
#     bot_id SERIAL,
#     ticker VARCHAR(10),
#     timestamp TIMESTAMP,
    
#     -- Performance Periods (percentages)
#XXXXX     one_hour_performance DECIMAL(6,2),  -- Increased precision for percentage
#XXXXX     two_hour_performance DECIMAL(6,2),
#     one_day_performance DECIMAL(6,2),
#     one_week_performance DECIMAL(6,2),
#     one_month_performance DECIMAL(6,2),
    
#     -- Core Metrics
#XXXXX     avg_win_rate DECIMAL(6,2),         -- Increased for more precise win rate
#     profit_per_second DECIMAL(10,4),   -- Increased decimal places for small per-second values
#XXXXX     total_pnl DECIMAL(12,2),          -- Large enough for significant PnL values
    
#     -- Trade Statistics
#XXXX     total_trades INTEGER,
#     trade_frequency INTEGER,           -- Trades per time period
#XXXXX     avg_profit_per_trade DECIMAL(10,2),
#XXXXX     profit_factor DECIMAL(10,2),      -- Ratio of profits to losses
    
#     -- Risk Metrics
#     avg_drawdown DECIMAL(6,2),
#     max_drawdown DECIMAL(6,2),
#     time_in_drawdown INTERVAL,
#     sharpe_ratio DECIMAL(8,4),        -- Standard format for Sharpe ratio
#     average_true_range DECIMAL(10,4),  -- ATR typically needs more precision
    
#     -- Execution Metrics
#     price_slippage DECIMAL(10,4),     -- Usually small values needing precision
#     time_slippage INTERVAL,
#     avg_trade_duration INTERVAL,
    
#     -- Model Scores
#     price_model_score DECIMAL(6,2),
#     volume_model_score DECIMAL(6,2),
#     price_wall_score DECIMAL(6,2),
    
#     -- Win Streaks (percentages)
#     win_streak_2 DECIMAL(6,2),
#     win_streak_3 DECIMAL(6,2),
#     win_streak_4 DECIMAL(6,2),
#     win_streak_5 DECIMAL(6,2),
    
#     -- Final Rankings
#     current_rank DECIMAL(6,2),
#     last_updated TIMESTAMP DEFAULT NOW()
# );
