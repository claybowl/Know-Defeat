import logging
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AIWeightAdjuster:
    """
    Machine learning-based weight adjustment system that optimizes variable weights
    based on historical performance data.
    """
    
    def __init__(self, db_pool):
        """Initialize with a database connection pool."""
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
    async def get_historical_performance(self, days=30):
        """Fetch historical bot performance data for training."""
        try:
            async with self.db_pool.acquire() as connection:
                # Get historical metrics for all bots
                metrics = await connection.fetch("""
                    SELECT 
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
                        price_model_score,
                        volume_model_score,
                        price_wall_score,
                        win_streak_2,
                        win_streak_3,
                        win_streak_4,
                        win_streak_5,
                        win_streak_6,
                        win_streak_7,
                        timestamp
                    FROM bot_metrics
                    WHERE timestamp >= $1
                    ORDER BY bot_id, timestamp
                """, datetime.now() - timedelta(days=days))
                
                # Convert to DataFrame
                metrics_df = pd.DataFrame(metrics)
                if metrics_df.empty:
                    return None
                
                # Get trade performance data
                trades = await connection.fetch("""
                    SELECT 
                        bot_id,
                        ticker,
                        trade_pnl,
                        timestamp
                    FROM sim_bot_trades
                    WHERE timestamp >= $1
                      AND trade_status = 'closed'
                    ORDER BY bot_id, timestamp
                """, datetime.now() - timedelta(days=days))
                
                trades_df = pd.DataFrame(trades)
                if trades_df.empty:
                    return None
                
                # Calculate the actual performance for each bot
                performance_df = trades_df.groupby('bot_id').agg({
                    'trade_pnl': ['sum', 'mean', 'count']
                }).reset_index()
                
                performance_df.columns = ['bot_id', 'total_pnl', 'avg_pnl', 'trade_count']
                
                # Combine metrics with performance
                return {
                    'metrics': metrics_df,
                    'performance': performance_df
                }
        except Exception as e:
            self.logger.error(f"Error fetching historical performance: {e}")
            return None
    
    async def train_model(self, historical_data):
        """
        Train a simple model to find optimal weights.
        
        This is a simplified version that uses correlation analysis.
        A more advanced implementation could use machine learning algorithms.
        """
        try:
            if not historical_data:
                self.logger.warning("No historical data available for training")
                return None
            
            metrics_df = historical_data['metrics']
            performance_df = historical_data['performance']
            
            # Group metrics by bot_id (use the most recent metrics for each bot)
            grouped_metrics = metrics_df.groupby('bot_id').last().reset_index()
            
            # Merge with performance data
            merged_df = pd.merge(grouped_metrics, performance_df, on='bot_id', how='inner')
            
            if merged_df.empty:
                self.logger.warning("No matching data between metrics and performance")
                return None
            
            # Define the target variable (we'll use total_pnl)
            target = 'total_pnl'
            
            # Define the metric columns to use for weight calculation
            metric_columns = [
                'one_hour_performance', 'two_hour_performance', 'one_day_performance',
                'one_week_performance', 'one_month_performance', 'avg_win_rate',
                'avg_drawdown', 'profit_per_second', 'price_model_score',
                'volume_model_score', 'price_wall_score', 'win_streak_2',
                'win_streak_3', 'win_streak_4', 'win_streak_5', 'win_streak_6',
                'win_streak_7'
            ]
            
            # Clean up data: replace NaN with 0
            for col in metric_columns:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(0)
            
            # Calculate correlation between each metric and the target
            correlations = {}
            for col in metric_columns:
                if col in merged_df.columns:
                    corr = merged_df[col].corr(merged_df[target])
                    correlations[col] = abs(corr) if not np.isnan(corr) else 0
            
            # If avg_drawdown is in the metrics, invert its correlation (lower is better)
            if 'avg_drawdown' in correlations:
                correlations['avg_drawdown'] = -correlations['avg_drawdown']
            
            # Normalize correlations to get weights
            total_corr = sum(abs(c) for c in correlations.values())
            weights = {k: (abs(v) / total_corr * 100) if total_corr > 0 else 0 for k, v in correlations.items()}
            
            # Ensure all weights are positive and sum to 100
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: (v / total_weight * 100) for k, v in weights.items()}
            else:
                # If all correlations are 0, assign equal weights
                weights = {k: (100 / len(weights)) for k in weights.keys()}
            
            return weights
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None
    
    async def adjust_weights(self):
        """
        Adjust variable weights based on historical performance analysis.
        """
        try:
            # Get historical data
            historical_data = await self.get_historical_performance()
            if not historical_data:
                self.logger.warning("No historical data available for weight adjustment")
                return False
            
            # Train model to get optimal weights
            optimal_weights = await self.train_model(historical_data)
            if not optimal_weights:
                self.logger.warning("Failed to determine optimal weights")
                return False
            
            # Update weights in database
            async with self.db_pool.acquire() as connection:
                # Ensure variable_weights table exists
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS variable_weights (
                        weight_id SERIAL PRIMARY KEY,
                        variable_name VARCHAR(50) NOT NULL UNIQUE,
                        weight DECIMAL(4,1) NOT NULL,
                        description TEXT,
                        last_updated TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Update weights
                for variable_name, weight in optimal_weights.items():
                    # Round weight to 1 decimal place
                    rounded_weight = round(weight, 1)
                    
                    # Get description
                    description = await self._get_variable_description(variable_name)
                    
                    # Update or insert weight
                    await connection.execute("""
                        INSERT INTO variable_weights 
                            (variable_name, weight, description, last_updated)
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (variable_name) 
                        DO UPDATE SET 
                            weight = $2,
                            description = $3,
                            last_updated = NOW()
                    """, variable_name, rounded_weight, description)
            
            self.logger.info("Successfully adjusted variable weights")
            return True
        except Exception as e:
            self.logger.error(f"Error adjusting weights: {e}")
            return False
    
    async def _get_variable_description(self, variable_name):
        """Get a human-readable description for a variable."""
        descriptions = {
            'one_hour_performance': 'Performance over the last hour',
            'two_hour_performance': 'Performance over the last two hours',
            'one_day_performance': 'Performance over the last day',
            'one_week_performance': 'Performance over the last week',
            'one_month_performance': 'Performance over the last month',
            'avg_win_rate': 'Average win rate of trades',
            'avg_drawdown': 'Average drawdown (lower is better)',
            'profit_per_second': 'Average profit per second of trading',
            'price_model_score': 'Score from price prediction model',
            'volume_model_score': 'Score from volume prediction model',
            'price_wall_score': 'Score based on order book price walls',
            'win_streak_2': 'Frequency of 2-trade win streaks',
            'win_streak_3': 'Frequency of 3-trade win streaks',
            'win_streak_4': 'Frequency of 4-trade win streaks',
            'win_streak_5': 'Frequency of 5-trade win streaks',
            'win_streak_6': 'Frequency of 6-trade win streaks',
            'win_streak_7': 'Frequency of 7-trade win streaks'
        }
        
        return descriptions.get(variable_name, f'Weight for {variable_name}')
    
    async def run_scheduled_adjustment(self, hours=24):
        """
        Run weight adjustment on a schedule.
        
        Args:
            hours: How often to run the adjustment (default: every 24 hours)
        """
        self.logger.info(f"Starting scheduled weight adjustment (every {hours} hours)")
        
        while True:
            try:
                await self.adjust_weights()
                self.logger.info(f"Next weight adjustment in {hours} hours")
                await asyncio.sleep(hours * 3600)  # Convert hours to seconds
            except Exception as e:
                self.logger.error(f"Error in scheduled adjustment: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retrying on error
