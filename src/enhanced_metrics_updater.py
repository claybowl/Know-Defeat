"""
Enhanced Metrics Updater

This module handles updating bot metrics in the database using the 
improved EnhancedMetricsCalculator for more accurate and comprehensive metrics.
"""

import asyncpg
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMetricsUpdater:
    """
    Enhanced metrics updater that uses the improved metrics calculator
    and handles new metrics storage.
    """
    
    def __init__(self, db_pool, metrics_calculator):
        """
        Initialize the enhanced metrics updater.
        
        Args:
            db_pool: Database connection pool
            metrics_calculator: Instance of EnhancedMetricsCalculator
        """
        self.db_pool = db_pool
        self.metrics_calculator = metrics_calculator
    
    def _limit_decimal_value(self, value: Union[float, Decimal, None], precision: int, scale: int) -> float:
        """
        Limit a value to fit within the specified precision and scale.
        
        Args:
            value: The numeric value to limit
            precision: Total number of digits (both sides of decimal point)
            scale: Number of digits after decimal point
            
        Returns:
            float: The value limited to fit within the specified precision/scale
        """
        try:
            if value is None:
                return 0.0
                
            # Convert to float if it's not already
            float_value = float(value) if not isinstance(value, float) else value
            
            # Calculate the maximum allowed value based on precision and scale
            digits_before_decimal = precision - scale
            max_value = 10 ** digits_before_decimal - 10 ** (-scale)
            min_value = -max_value
            
            # Limit the value to the allowed range
            limited_value = max(min(float_value, max_value), min_value)
            
            # Round to the specified scale
            return round(limited_value, scale)
        except (TypeError, ValueError):
            # Return 0 if the value can't be converted
            return 0.0
    
    async def _ensure_bot_metrics_table(self, connection):
        """
        Ensure the bot_metrics table exists with all required columns.
        
        Args:
            connection: Database connection
        """
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
                win_streak_6 DECIMAL(6,2),
                win_streak_7 DECIMAL(6,2),
                
                -- Enhanced Risk Metrics
                sortino_ratio DECIMAL(12,6),
                calmar_ratio DECIMAL(12,6),
                r_multiple DECIMAL(12,6),
                max_drawdown_duration DECIMAL(20,4),
                recovery_factor DECIMAL(12,6),
                drawdown_percent DECIMAL(8,4),
                
                -- Final Rankings
                current_rank DECIMAL(6,2),
                last_updated TIMESTAMP DEFAULT NOW(),
                
                -- Constraints
                PRIMARY KEY (bot_id, timestamp)
            )
        """)
        
        # Create index for faster lookups
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_bot_metrics_bot_id_timestamp 
            ON bot_metrics (bot_id, timestamp DESC)
        """)
    
    async def get_algorithm_id(self, connection, bot_id: int) -> int:
        """
        Get the algorithm ID for a bot, falling back to bot_id if not found.
        
        Args:
            connection: Database connection
            bot_id: Bot ID
            
        Returns:
            int: Algorithm ID
        """
        try:
            # Try to get algorithm_id from sim_bots table
            algo_id = await connection.fetchval("""
                SELECT algorithm_id FROM sim_bots WHERE bot_id = $1
            """, bot_id)
            
            # Fall back to bot_id if not found
            return algo_id if algo_id is not None else bot_id
        except Exception as e:
            logger.warning(f"Could not retrieve algorithm_id for bot {bot_id}: {e}")
            return bot_id
    
    async def update_bot_metrics(self, bot_id: int, ticker: str) -> bool:
        """
        Update metrics for a bot using the enhanced metrics calculator.
        
        Args:
            bot_id: Bot ID
            ticker: Ticker symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate inputs
            if not bot_id or not isinstance(bot_id, int):
                logger.error(f"Invalid bot_id: {bot_id}")
                return False
                
            if not ticker or not isinstance(ticker, str):
                logger.error(f"Invalid ticker: {ticker}")
                return False
            
            # Get all metrics at once using the enhanced calculator
            metrics = await self.metrics_calculator.calculate_all_metrics(bot_id, ticker)
            
            if not metrics:
                logger.error(f"Failed to calculate metrics for bot {bot_id}, ticker {ticker}")
                return False
            
            # Attempt to insert metrics into the database
            try:
                async with self.db_pool.acquire() as connection:
                    # Ensure the table exists with all required columns
                    await self._ensure_bot_metrics_table(connection)
                    
                    # Get algorithm ID
                    algo_id = await self.get_algorithm_id(connection, bot_id)
                    
                    # Current timestamp for all metrics
                    now = datetime.now()
                    
                    # Prepare key-value pairs for insert
                    columns = ["bot_id", "ticker", "algo_id", "timestamp"]
                    values = [bot_id, ticker, algo_id, now]
                    
                    # Add all metrics
                    for key, value in metrics.items():
                        # Skip any None values or dict values
                        if value is None or isinstance(value, dict):
                            continue
                            
                        columns.append(key)
                        
                        # Handle different data types
                        if key in ['avg_trade_duration', 'time_in_drawdown', 'time_slippage']:
                            # For interval types, store as seconds
                            if isinstance(value, (int, float, Decimal)):
                                values.append(f"{value} seconds")
                            else:
                                values.append(value)
                        else:
                            # For numeric types, apply appropriate limits
                            if key in ['one_hour_performance', 'two_hour_performance', 
                                    'one_day_performance', 'one_week_performance', 
                                    'one_month_performance', 'avg_win_rate',
                                    'avg_drawdown', 'max_drawdown',
                                    'price_model_score', 'volume_model_score', 
                                    'price_wall_score',
                                    'win_streak_2', 'win_streak_3', 'win_streak_4', 
                                    'win_streak_5', 'win_streak_6', 'win_streak_7',
                                    'drawdown_percent']:
                                limited_value = self._limit_decimal_value(value, 6, 2)
                            elif key in ['sharpe_ratio', 'sortino_ratio', 
                                      'calmar_ratio', 'r_multiple', 
                                      'recovery_factor']:
                                limited_value = self._limit_decimal_value(value, 12, 6)
                            elif key in ['profit_per_second', 'average_true_range',
                                      'price_slippage']:
                                limited_value = self._limit_decimal_value(value, 10, 4)
                            elif key in ['profit_factor', 'avg_profit_per_trade']:
                                limited_value = self._limit_decimal_value(value, 10, 2)
                            elif key == 'total_pnl':
                                limited_value = self._limit_decimal_value(value, 12, 2)
                            elif key == 'max_drawdown_duration':
                                limited_value = self._limit_decimal_value(value, 20, 4)
                            elif key in ['total_trades', 'trade_frequency']:
                                limited_value = int(value)
                            else:
                                limited_value = value
                                
                            values.append(limited_value)
                    
                    # Build the dynamic INSERT statement
                    placeholders = [f"${i+1}" for i in range(len(values))]
                    insert_query = f"""
                        INSERT INTO bot_metrics ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    
                    # Execute the insert
                    await connection.execute(insert_query, *values)
                    
                    logger.info(f"Updated metrics for bot {bot_id}, ticker {ticker}, algorithm {algo_id}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error updating metrics for bot {bot_id}, ticker {ticker}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in update_bot_metrics for bot {bot_id}, ticker {ticker}: {e}")
            return False