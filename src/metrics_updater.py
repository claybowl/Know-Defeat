import asyncpg
import logging

class MetricsUpdater:
    def __init__(self, db_pool, metrics_calculator):
        self.db_pool = db_pool
        self.metrics_calculator = metrics_calculator

    async def update_bot_metrics(self, bot_id, ticker):
        one_hour_perf = await self.metrics_calculator.calculate_one_hour_performance(bot_id, ticker)
        avg_win_rate = await self.metrics_calculator.calculate_avg_win_rate(bot_id, ticker)
        max_drawdown = await self.metrics_calculator.calculate_max_drawdown(bot_id, ticker)

        async with self.db_pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO bot_metrics (bot_id, ticker, one_hour_performance, avg_win_rate)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (bot_id, ticker) DO UPDATE
                SET one_hour_performance = $3, avg_win_rate = $4;
            """, bot_id, ticker, one_hour_perf, avg_win_rate)
        
        logging.info(f"Updated metrics for bot {bot_id}, ticker {ticker}")
