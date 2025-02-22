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

    # Add more metric calculation functions as needed
    # For example:
    # async def calculate_max_drawdown(self, bot_id, ticker):
    #     ...
    #     return ...
