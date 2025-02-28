import asyncpg
import logging

class MetricsUpdater:
    def __init__(self, db_pool, metrics_calculator):
        self.db_pool = db_pool
        self.metrics_calculator = metrics_calculator

    async def update_bot_metrics(self, bot_id, ticker):
        # Calculate basic metrics (you already have these)
        one_hour_perf = await self.metrics_calculator.calculate_one_hour_performance(bot_id, ticker)
        avg_win_rate = await self.metrics_calculator.calculate_avg_win_rate(bot_id, ticker)
        
        # Calculate additional metrics needed for ranking
        two_hour_perf = await self.metrics_calculator.calculate_two_hour_performance(bot_id, ticker)
        one_day_perf = await self.metrics_calculator.calculate_one_day_performance(bot_id, ticker)
        one_week_perf = await self.metrics_calculator.calculate_one_week_performance(bot_id, ticker)
        one_month_perf = await self.metrics_calculator.calculate_one_month_performance(bot_id, ticker)
        profit_per_second = await self.metrics_calculator.calculate_profit_per_second(bot_id, ticker)
        drawdown_info = await self.metrics_calculator.calculate_drawdowns(bot_id, ticker)
        
        # Get algorithm ID (you may need to adjust this to get the correct algo_id)
        algo_id = 1  # Default value, replace with actual logic to get algo_id
        
        async with self.db_pool.acquire() as connection:
            # Update metrics in database with all needed values
            await connection.execute("""
                INSERT INTO bot_metrics (
                    bot_id, 
                    ticker, 
                    algorithm_id,
                    one_hour_performance, 
                    two_hour_performance,
                    one_day_performance,
                    one_week_performance,
                    one_month_performance,
                    avg_win_rate, 
                    avg_drawdown,
                    profit_per_second,
                    timestamp
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                ON CONFLICT (bot_id, algorithm_id) DO UPDATE
                SET 
                    one_hour_performance = $4, 
                    two_hour_performance = $5,
                    one_day_performance = $6,
                    one_week_performance = $7,
                    one_month_performance = $8,
                    avg_win_rate = $9, 
                    avg_drawdown = $10,
                    profit_per_second = $11,
                    timestamp = NOW();
            """, 
            bot_id, ticker, algo_id, 
            one_hour_perf, two_hour_perf, one_day_perf, one_week_perf, one_month_perf,
            avg_win_rate, drawdown_info['avg_drawdown'], profit_per_second)
        
        # Calculate and update win streaks
        await self.metrics_calculator.calculate_and_insert_win_streaks(bot_id, algo_id)
        
        logging.info(f"Updated metrics for bot {bot_id}, ticker {ticker}")
