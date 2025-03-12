
"""
Mean Reversion Algorithm

This algorithm implements a basic mean reversion strategy for tick data.
"""

import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class Mean_reversion_algorithmAlgorithm:
    def __init__(self, direction, parameters):
        """Initialize with direction and parameters from YAML config"""
        self.direction = direction
        
        # Extract parameters with defaults
        self.lookback_periods = parameters.get('lookback_periods', 100)
        self.std_dev_threshold = parameters.get('std_dev_threshold', 2.0)
        self.profit_target_pct = parameters.get('profit_target_pct', 0.5)
        self.stop_loss_pct = parameters.get('stop_loss_pct', 0.3)
        
        logger.info(f"Initialized {direction} mean reversion algorithm with "
                   f"lookback={self.lookback_periods}, "
                   f"threshold={self.std_dev_threshold}σ")
        
    def check_entry_conditions(self, ticks_df):
        """Check if entry conditions are met based on tick data"""
        if ticks_df is None or len(ticks_df) < self.lookback_periods:
            return False
            
        try:
            # Calculate mean and standard deviation
            mean_price = ticks_df['price'].mean()
            std_dev = ticks_df['price'].std()
            current_price = float(ticks_df['price'].iloc[0])
            
            # Calculate z-score (how many standard deviations from mean)
            z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
            
            logger.info(f"Mean: {mean_price:.4f}, StdDev: {std_dev:.4f}, "
                       f"Current: {current_price:.4f}, Z-score: {z_score:.2f}")
            
            # For LONG: Enter when price is significantly below mean (negative z-score)
            # For SHORT: Enter when price is significantly above mean (positive z-score)
            if self.direction == 'LONG':
                return z_score <= -self.std_dev_threshold
            else:  # SHORT
                return z_score >= self.std_dev_threshold
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """Check if exit conditions are met"""
        try:
            entry_price = position_data.get('entry_price', current_price)
            
            # Calculate profit/loss percentage
            if self.direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price * 100
                # Exit when profit target hit or stop loss hit
                return (pnl_pct >= self.profit_target_pct or 
                        pnl_pct <= -self.stop_loss_pct)
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price * 100
                # Exit when profit target hit or stop loss hit
                return (pnl_pct >= self.profit_target_pct or 
                        pnl_pct <= -self.stop_loss_pct)
                
        except Exception as e:
            logger.error(f"Error in check_exit_conditions: {e}")
            return False
