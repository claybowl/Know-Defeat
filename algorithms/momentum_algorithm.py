"""
Momentum Algorithm Implementation

This module implements a momentum-based trading algorithm that can be used
for both long and short strategies.
"""

import pandas as pd
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class Momentum_algorithmAlgorithm:
    """
    Momentum algorithm for detecting price trends and generating trade signals.
    """

    def __init__(self, direction, parameters):
        """
        Initialize the momentum algorithm.
        
        Args:
            direction: 'LONG' or 'SHORT' - trade direction
            parameters: Dictionary of algorithm parameters from YAML config
        """
        self.direction = direction
        self.lookback_seconds = parameters.get('lookback_seconds', 60)
        self.momentum_threshold = parameters.get('momentum_threshold', 0.001)
        self.confirmation_seconds = parameters.get('confirmation_seconds', 15)
        self.use_volume = parameters.get('use_volume', False)
        self.min_bars = parameters.get('min_bars', 2)

        logger.info(
            f"Initialized {direction} momentum algorithm with "
            f"lookback={self.lookback_seconds}s, "
            f"threshold={self.momentum_threshold}, "
            f"confirmation={self.confirmation_seconds}s"
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
            
            # Skip if price is invalid
            if ticks_df['price'].iloc[0] <= 0:
                logger.warning(f"Skipping invalid current price: {ticks_df['price'].iloc[0]}")
                return False
            
            # Extract current price and price from lookback period
            # Convert all prices to float for consistent calculations
            current_price = float(ticks_df['price'].iloc[0])
            oldest_price = float(ticks_df['price'].iloc[-1])
            
            # Calculate the percentage change
            pct_change = (current_price - oldest_price) / oldest_price
            
            # Get confirmation price (from N seconds ago)
            latest_time = ticks_df['timestamp'].iloc[0]
            cutoff_time = latest_time - timedelta(seconds=self.confirmation_seconds)
            confirm_ticks = ticks_df[ticks_df['timestamp'] >= cutoff_time]
            
            if len(confirm_ticks) == 0:
                logger.debug(f"No ticks found for confirmation period ({self.confirmation_seconds}s)")
                return False
                
            confirmation_price = float(confirm_ticks['price'].iloc[-1])
            
            # Log key metrics
            logger.info(
                "Current price: %.4f, %ds ago: %.4f, Change: %.4f%%, Confirmation price: %.4f",
                current_price,
                self.lookback_seconds,
                oldest_price,
                pct_change * 100,  # Convert to percentage
                confirmation_price
            )
            
            # Check volume if required
            volume_condition = True
            if self.use_volume and 'volume' in ticks_df.columns:
                # Implement volume condition here
                pass
            
            # Check conditions based on direction
            if self.direction == 'LONG':
                # For long entries:
                # 1. Current price must be higher than lookback price by threshold
                # 2. Current price must be at least as high as confirmation price
                return (pct_change >= self.momentum_threshold and 
                        current_price >= confirmation_price and
                        volume_condition)
            else:  # SHORT
                # For short entries:
                # 1. Current price must be lower than lookback price by threshold
                # 2. Current price must be at most as high as confirmation price
                return (pct_change <= -self.momentum_threshold and 
                        current_price <= confirmation_price and
                        volume_condition)
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """
        Check if exit conditions are met based on current price and position data.
        
        Args:
            current_price: Current price of the asset
            position_data: Dictionary containing position information including:
                - entry_price: Price at which the position was entered
                - extreme_price: Highest (for LONG) or lowest (for SHORT) price since entry
                - trailing_stop_pct: Trailing stop percentage
                
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
