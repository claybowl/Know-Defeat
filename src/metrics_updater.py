import asyncpg
import logging
from decimal import Decimal

class MetricsUpdater:
    def __init__(self, db_pool, metrics_calculator):
        self.db_pool = db_pool
        self.metrics_calculator = metrics_calculator

    async def update_bot_metrics(self, bot_id, ticker):
        try:
            # Validate inputs
            if not bot_id or not isinstance(bot_id, int):
                logging.error(f"Invalid bot_id: {bot_id}")
                return False
                
            if not ticker or not isinstance(ticker, str):
                logging.error(f"Invalid ticker: {ticker}")
                return False
            
            # Simply use bot_id as algo_id
            algo_id = bot_id
            
            try:
                # Calculate basic metrics
                one_hour_perf = await self.metrics_calculator.calculate_one_hour_performance(bot_id, ticker)
                avg_win_rate = await self.metrics_calculator.calculate_avg_win_rate(bot_id, ticker)
                
                # Calculate additional metrics needed for ranking
                two_hour_perf = await self.metrics_calculator.calculate_two_hour_performance(bot_id, algo_id)
                one_day_perf = await self.metrics_calculator.calculate_one_day_performance(bot_id, algo_id)
                one_week_perf = await self.metrics_calculator.calculate_one_week_performance(bot_id, algo_id)
                one_month_perf = await self.metrics_calculator.calculate_one_month_performance(bot_id, algo_id)
                profit_per_second = await self.metrics_calculator.calculate_profit_per_second(bot_id, algo_id)
                total_pnl = await self.metrics_calculator.calculate_total_pnl(bot_id, algo_id)
                total_trades = await self.metrics_calculator.calculate_total_trades(bot_id, algo_id)
                avg_profit_per_trade = await self.metrics_calculator.calculate_avg_profit_per_trade(bot_id, algo_id)
                
                # Calculate drawdown metrics
                drawdown_info = await self.metrics_calculator.calculate_drawdowns(bot_id, algo_id)
                
                # Calculate model scores
                price_model_score = await self.metrics_calculator.calculate_price_model_score(bot_id, algo_id)
                volume_model_score = await self.metrics_calculator.calculate_volume_model_score(bot_id, algo_id)
                price_wall_score = await self.metrics_calculator.calculate_price_wall_score(bot_id, algo_id)
            except Exception as e:
                logging.error(f"Error calculating metrics for bot {bot_id}, ticker {ticker}: {e}")
                return False
            
            # Ensure all values are properly converted to the right type
            try:
                # Convert all potential Decimal values to float
                if isinstance(one_hour_perf, Decimal):
                    one_hour_perf = float(one_hour_perf)
                if isinstance(two_hour_perf, Decimal):
                    two_hour_perf = float(two_hour_perf)
                if isinstance(one_day_perf, Decimal):
                    one_day_perf = float(one_day_perf)
                if isinstance(one_week_perf, Decimal):
                    one_week_perf = float(one_week_perf)
                if isinstance(one_month_perf, Decimal):
                    one_month_perf = float(one_month_perf)
                if isinstance(avg_win_rate, Decimal):
                    avg_win_rate = float(avg_win_rate)
                if isinstance(profit_per_second, Decimal):
                    profit_per_second = float(profit_per_second)
                if isinstance(total_pnl, Decimal):
                    total_pnl = float(total_pnl)
                if isinstance(avg_profit_per_trade, Decimal):
                    avg_profit_per_trade = float(avg_profit_per_trade)
                
                # Handle potential Decimal values in drawdown_info
                for key in ['avg_drawdown', 'max_drawdown']:
                    if key in drawdown_info and isinstance(drawdown_info[key], Decimal):
                        drawdown_info[key] = float(drawdown_info[key])
                        
                if isinstance(price_model_score, Decimal):
                    price_model_score = float(price_model_score)
                if isinstance(volume_model_score, Decimal):
                    volume_model_score = float(volume_model_score)
                if isinstance(price_wall_score, Decimal):
                    price_wall_score = float(price_wall_score)
            except Exception as e:
                logging.error(f"Error converting metric types for bot {bot_id}: {e}")
                # Continue with the values we have
            
            try:
                async with self.db_pool.acquire() as connection:
                    # First check if the bot_metrics table exists, create if not
                    await connection.execute("""
                        CREATE TABLE IF NOT EXISTS bot_metrics (
                            -- Identifiers
                            bot_id INTEGER,
                            ticker VARCHAR(10),
                            algo_id INTEGER,
                            timestamp TIMESTAMP,
                            
                            -- Performance Periods (percentages)
                            one_hour_performance DECIMAL(6,2),
                            two_hour_performance DECIMAL(6,2),
                            one_day_performance DECIMAL(6,2),
                            one_week_performance DECIMAL(6,2),
                            one_month_performance DECIMAL(6,2),
                            
                            -- Core Metrics
                            avg_win_rate DECIMAL(6,2),
                            profit_per_second DECIMAL(10,4),
                            total_pnl DECIMAL(12,2),
                            
                            -- Trade Statistics
                            total_trades INTEGER,
                            trade_frequency INTEGER,
                            avg_profit_per_trade DECIMAL(10,2),
                            profit_factor DECIMAL(10,2),
                            
                            -- Risk Metrics
                            avg_drawdown DECIMAL(6,2),
                            max_drawdown DECIMAL(6,2),
                            time_in_drawdown INTERVAL,
                            sharpe_ratio DECIMAL(8,4),
                            average_true_range DECIMAL(10,4),
                            
                            -- Execution Metrics
                            price_slippage DECIMAL(10,4),
                            time_slippage INTERVAL,
                            avg_trade_duration INTERVAL,
                            
                            -- Model Scores
                            price_model_score DECIMAL(6,2),
                            volume_model_score DECIMAL(6,2),
                            price_wall_score DECIMAL(6,2),
                            
                            -- Win Streaks (percentages)
                            win_streak_2 DECIMAL(6,2),
                            win_streak_3 DECIMAL(6,2),
                            win_streak_4 DECIMAL(6,2),
                            win_streak_5 DECIMAL(6,2),
                            
                            -- Final Rankings
                            current_rank DECIMAL(6,2),
                            last_updated TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # Update all metrics in a single transaction
                    await connection.execute("""
                        INSERT INTO bot_metrics (
                            bot_id, 
                            ticker, 
                            algo_id,
                            timestamp,
                            one_hour_performance, 
                            two_hour_performance,
                            one_day_performance,
                            one_week_performance,
                            one_month_performance,
                            avg_win_rate, 
                            profit_per_second,
                            total_pnl,
                            total_trades,
                            avg_profit_per_trade,
                            avg_drawdown,
                            max_drawdown,
                            price_model_score,
                            volume_model_score,
                            price_wall_score
                        )
                        VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                    """, 
                    bot_id, 
                    ticker, 
                    algo_id, 
                    float(one_hour_perf), 
                    float(two_hour_perf), 
                    float(one_day_perf), 
                    float(one_week_perf), 
                    float(one_month_perf),
                    float(avg_win_rate), 
                    float(profit_per_second), 
                    float(total_pnl), 
                    int(total_trades), 
                    float(avg_profit_per_trade),
                    float(drawdown_info['avg_drawdown']), 
                    float(drawdown_info['max_drawdown']),
                    float(price_model_score), 
                    float(volume_model_score), 
                    float(price_wall_score))
                
                # Calculate and update win streaks separately
                win_streaks = await self.metrics_calculator.calculate_and_insert_win_streaks(bot_id, algo_id)
                
                logging.info(f"Updated metrics for bot {bot_id}, ticker {ticker}, algorithm {algo_id}")
                return True
                
            except Exception as e:
                logging.error(f"Error updating metrics for bot {bot_id}, ticker {ticker}: {e}")
                return False
                
        except Exception as e:
            logging.error(f"Error in update_bot_metrics for bot {bot_id}, ticker {ticker}: {e}")
            return False
