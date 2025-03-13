"""
Breakout Algorithm

This algorithm identifies and trades price breakouts from consolidation periods.
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class BreakoutAlgorithm:
    def __init__(self, direction, parameters):
        """Initialize with direction and parameters from YAML config"""
        self.direction = direction
        
        # Extract parameters with defaults
        self.lookback_seconds = parameters.get('lookback_seconds', 300)
        self.consolidation_threshold = parameters.get('consolidation_threshold', 0.0015)  # 0.15%
        self.breakout_threshold = parameters.get('breakout_threshold', 0.002)  # 0.2%
        self.confirmation_seconds = parameters.get('confirmation_seconds', 10)
        self.min_bars = parameters.get('min_bars', 10)
        
        logger.info(
            f"Initialized {direction} breakout algorithm with "
            f"lookback={self.lookback_seconds}s, "
            f"consolidation_threshold={self.consolidation_threshold}, "
            f"breakout_threshold={self.breakout_threshold}"
        )
    
    def check_entry_conditions(self, ticks_df):
        """
        Check if entry conditions are met based on current price data.
        
        Args:
            ticks_df: DataFrame containing tick data with 'timestamp' and 'price' columns
            
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        try:
            # Skip if we don't have enough data
            if ticks_df is None or len(ticks_df) < self.min_bars:
                return False
            
            # Convert all prices to float for consistent calculations
            current_price = float(ticks_df['price'].iloc[0])
            
            # Calculate price range during the lookback period
            prices = ticks_df['price'].values.astype(float)
            consolidation_period = prices[1:]  # exclude current price
            
            if len(consolidation_period) < 3:
                return False
                
            price_range = np.max(consolidation_period) - np.min(consolidation_period)
            avg_price = np.mean(consolidation_period)
            range_pct = price_range / avg_price
            
            # Get confirmation price from N seconds ago
            latest_time = ticks_df['timestamp'].iloc[0]
            cutoff_time = latest_time - timedelta(seconds=self.confirmation_seconds)
            confirm_ticks = ticks_df[ticks_df['timestamp'] >= cutoff_time]
            
            if len(confirm_ticks) < 2:
                return False
                
            confirmation_price = float(confirm_ticks['price'].iloc[-1])
            
            # Calculate breakout percentage
            if self.direction == 'LONG':
                # For long entries, compare to previous high
                previous_high = np.max(consolidation_period)
                breakout_pct = (current_price - previous_high) / previous_high
                
                # Long entry conditions:
                # 1. Price range during consolidation period is small (tight range)
                # 2. Current price breaks above the previous high by threshold
                # 3. Confirmation price is below current price (upward momentum)
                is_tight_range = range_pct <= self.consolidation_threshold
                is_breakout = breakout_pct >= self.breakout_threshold
                is_confirmed = current_price >= confirmation_price
                
                logger.info(
                    f"LONG: Current: {current_price:.4f}, Prev High: {previous_high:.4f}, "
                    f"Breakout: {breakout_pct*100:.2f}%, Range: {range_pct*100:.2f}%, "
                    f"Tight range: {is_tight_range}, Breakout: {is_breakout}, Confirmed: {is_confirmed}"
                )
                
                return is_tight_range and is_breakout and is_confirmed
                
            else:  # SHORT
                # For short entries, compare to previous low
                previous_low = np.min(consolidation_period)
                breakout_pct = (previous_low - current_price) / previous_low
                
                # Short entry conditions:
                # 1. Price range during consolidation period is small (tight range)
                # 2. Current price breaks below the previous low by threshold
                # 3. Confirmation price is above current price (downward momentum)
                is_tight_range = range_pct <= self.consolidation_threshold
                is_breakout = breakout_pct >= self.breakout_threshold
                is_confirmed = current_price <= confirmation_price
                
                logger.info(
                    f"SHORT: Current: {current_price:.4f}, Prev Low: {previous_low:.4f}, "
                    f"Breakout: {breakout_pct*100:.2f}%, Range: {range_pct*100:.2f}%, "
                    f"Tight range: {is_tight_range}, Breakout: {is_breakout}, Confirmed: {is_confirmed}"
                )
                
                return is_tight_range and is_breakout and is_confirmed
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """
        Check if exit conditions are met based on current price and position data.
        
        Args:
            current_price: Current price of the asset
            position_data: Dictionary containing position information
                
        Returns:
            bool: True if exit conditions are met, False otherwise
        """
        try:
            # Extract position data
            entry_price = position_data.get('entry_price', current_price)
            extreme_price = position_data.get('extreme_price', current_price)
            trailing_stop_pct = position_data.get('trailing_stop_pct', 0.002)
            
            # Update extreme price
            if self.direction == 'LONG':
                # For long positions, track highest price
                if current_price > extreme_price:
                    position_data['extreme_price'] = current_price
                    extreme_price = current_price
                
                # Calculate stop price
                stop_price = extreme_price * (1 - trailing_stop_pct)
                
                # Exit if price falls below stop price
                if current_price <= stop_price:
                    logger.info(
                        f"Long exit triggered: Current {current_price:.4f} below "
                        f"stop at {stop_price:.4f} (highest: {extreme_price:.4f})"
                    )
                    return True
                
            else:  # SHORT
                # For short positions, track lowest price
                if current_price < extreme_price:
                    position_data['extreme_price'] = current_price
                    extreme_price = current_price
                
                # Calculate stop price
                stop_price = extreme_price * (1 + trailing_stop_pct)
                
                # Exit if price rises above stop price
                if current_price >= stop_price:
                    logger.info(
                        f"Short exit triggered: Current {current_price:.4f} above "
                        f"stop at {stop_price:.4f} (lowest: {extreme_price:.4f})"
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in check_exit_conditions: {e}")
            return False