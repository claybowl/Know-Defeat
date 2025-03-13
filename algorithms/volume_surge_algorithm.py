
"""
Volume Surge Algorithm

This algorithm identifies unusual volume spikes combined with
significant price movements as a sign of potential trend continuation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class Volume_surgeAlgorithm:
    def __init__(self, direction, parameters):
        """Initialize with direction and parameters from YAML config"""
        self.direction = direction
        
        # Extract parameters with defaults
        self.lookback_ticks = parameters.get('lookback_ticks', 100)
        self.volume_threshold = parameters.get('volume_threshold', 2.5)  # Volume must be X times average
        self.price_move_threshold = parameters.get('price_move_threshold', 0.3)  # Min % price move
        self.confirmation_ticks = parameters.get('confirmation_ticks', 5)  # Ticks to confirm direction
        self.profit_take_r = parameters.get('profit_take_r', 2.0)  # Target as multiple of initial risk
        self.stop_loss_pct = parameters.get('stop_loss_pct', 0.5)  # Stop loss %
        
        logger.info(f"Initialized {direction} volume surge algorithm with "
                   f"lookback={self.lookback_ticks}, "
                   f"volume_threshold={self.volume_threshold}x, "
                   f"price_move={self.price_move_threshold}%")
        
    def check_entry_conditions(self, ticks_df):
        """Check if entry conditions are met based on tick data"""
        if ticks_df is None or len(ticks_df) < self.lookback_ticks:
            return False
            
        try:
            if 'volume' not in ticks_df.columns:
                logger.warning("Volume data not available for volume surge algorithm")
                return False
                
            # Calculate average volume over lookback period
            avg_volume = ticks_df['volume'].iloc[10:].mean()  # Skip most recent for avg calculation
            
            # Get recent volume and price data
            recent_volume = ticks_df['volume'].iloc[:10].mean()
            current_price = float(ticks_df['price'].iloc[0])
            price_5_ticks_ago = float(ticks_df['price'].iloc[self.confirmation_ticks])
            price_lookback = float(ticks_df['price'].iloc[-1])
            
            # Calculate price changes
            recent_pct_change = (current_price - price_5_ticks_ago) / price_5_ticks_ago * 100
            overall_pct_change = (current_price - price_lookback) / price_lookback * 100
            
            # Check for volume surge
            volume_surge = recent_volume > (avg_volume * self.volume_threshold)
            
            logger.info(f"Recent volume: {recent_volume:.1f} vs avg: {avg_volume:.1f} (surge: {volume_surge}), "
                       f"Recent price change: {recent_pct_change:.2f}%, "
                       f"Overall change: {overall_pct_change:.2f}%")
            
            # Entry conditions based on direction
            if self.direction == 'LONG':
                # For long entry: volume surge + positive price movement in short and longer timeframes
                price_condition = (recent_pct_change > 0 and 
                                  overall_pct_change > self.price_move_threshold)
                
                if volume_surge and price_condition:
                    logger.info(f"LONG volume surge signal at {current_price:.4f}")
                    return True
                    
            else:  # SHORT
                # For short entry: volume surge + negative price movement in short and longer timeframes
                price_condition = (recent_pct_change < 0 and 
                                  overall_pct_change < -self.price_move_threshold)
                
                if volume_surge and price_condition:
                    logger.info(f"SHORT volume surge signal at {current_price:.4f}")
                    return True
                    
            return False
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """Check if exit conditions are met"""
        try:
            entry_price = position_data.get('entry_price', current_price)
            
            # Calculate risk (initial distance to stop)
            risk = entry_price * (self.stop_loss_pct / 100)
            
            # Calculate target and stop levels
            if self.direction == 'LONG':
                target_price = entry_price + (risk * self.profit_take_r)
                stop_price = entry_price - risk
                
                # Check for exit conditions
                hit_target = current_price >= target_price
                hit_stop = current_price <= stop_price
                
                if hit_target:
                    logger.info(f"LONG target hit: {current_price:.4f} >= {target_price:.4f}")
                if hit_stop:
                    logger.info(f"LONG stop hit: {current_price:.4f} <= {stop_price:.4f}")
                    
                return hit_target or hit_stop
                
            else:  # SHORT
                target_price = entry_price - (risk * self.profit_take_r)
                stop_price = entry_price + risk
                
                # Check for exit conditions
                hit_target = current_price <= target_price
                hit_stop = current_price >= stop_price
                
                if hit_target:
                    logger.info(f"SHORT target hit: {current_price:.4f} <= {target_price:.4f}")
                if hit_stop:
                    logger.info(f"SHORT stop hit: {current_price:.4f} >= {stop_price:.4f}")
                    
                return hit_target or hit_stop
                
        except Exception as e:
            logger.error(f"Error in check_exit_conditions: {e}")
            return False
