"""
Minute-Based Momentum Algorithm Implementation

This module implements an enhanced momentum-based trading algorithm that uses
both tick data and minute OHLC data for more robust entry/exit decisions.
This version is specifically tailored to capture the logic found in the
COIN and TSLA2 bots.
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Minute_momentum_algorithmAlgorithm:
    """
    Enhanced momentum algorithm that uses both tick-level data and minute OHLC data
    for detecting price trends and generating trade signals.
    """
    
    def __init__(self, direction, parameters):
        """
        Initialize the minute-based momentum algorithm.
        
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
        self.use_minute_ohlc = parameters.get('use_minute_ohlc', True)
        self.minute_confirmation = parameters.get('minute_confirmation', True)
        
        # Minute OHLC tracking
        self.last_minute_open = None
        self.last_minute_high = None
        self.last_minute_low = None
        self.last_minute_close = None
        self.current_minute = None
        
        logger.info(
            f"Initialized {direction} minute momentum algorithm with "
            f"lookback={self.lookback_seconds}s, "
            f"threshold={self.momentum_threshold}, "
            f"confirmation={self.confirmation_seconds}s, "
            f"minute_ohlc={self.use_minute_ohlc}"
        )
    
    def update_minute_data(self, timestamp, price):
        """
        Update the one-minute OHLC data based on the current price.
        
        Args:
            timestamp: Timestamp of the current tick
            price: Current price
        """
        # Extract the current minute (truncating seconds)
        current_minute = timestamp.replace(second=0, microsecond=0)
        
        # If this is a new minute, reset the OHLC values
        if self.current_minute != current_minute:
            logger.debug(f"Starting new minute: {current_minute}")
            self.current_minute = current_minute
            self.last_minute_open = price
            self.last_minute_high = price
            self.last_minute_low = price
        else:
            # Update high and low prices for the current minute
            if price > self.last_minute_high:
                self.last_minute_high = price
            if price < self.last_minute_low:
                self.last_minute_low = price
        
        # Always update the close price
        self.last_minute_close = price

    def check_minute_conditions(self):
        """
        Check if the minute-based conditions are met for entry.
        
        Returns:
            bool: True if minute conditions are met, False otherwise
        """
        if not self.use_minute_ohlc or not self.minute_confirmation:
            return True
            
        if None in (self.last_minute_open, self.last_minute_close):
            logger.debug("Minute OHLC data not yet available")
            return False
            
        # Check minute conditions based on direction
        if self.direction == 'LONG':
            # For long entries, want to see close > open (bullish candle)
            minute_condition = self.last_minute_close > self.last_minute_open
        else:  # SHORT
            # For short entries, want to see close < open (bearish candle)
            minute_condition = self.last_minute_close < self.last_minute_open
            
        if minute_condition:
            logger.info(
                f"Minute condition met: Open={self.last_minute_open:.4f}, "
                f"Close={self.last_minute_close:.4f}"
            )
        else:
            logger.debug(
                f"Minute condition not met: Open={self.last_minute_open:.4f}, "
                f"Close={self.last_minute_close:.4f}"
            )
            
        return minute_condition
    
    def check_entry_conditions(self, ticks_df):
        """
        Check if entry conditions are met based on current price data.
        
        Args:
            ticks_df: DataFrame containing tick data with 'timestamp' and 'price' columns
            
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        if ticks_df is None or len(ticks_df) < self.min_bars:
            logger.debug("Insufficient data for analysis")
            return False
        
        try:
            # Extract current price and price from lookback period
            current_price = float(ticks_df['price'].iloc[0])
            oldest_price = float(ticks_df['price'].iloc[-1])
            latest_timestamp = ticks_df['timestamp'].iloc[0]
            
            # Update the minute OHLC data
            self.update_minute_data(latest_timestamp, current_price)
            
            # Calculate the percentage change
            pct_change = (current_price - oldest_price) / oldest_price
            
            # Get confirmation price (from N seconds ago)
            cutoff_time = latest_timestamp - timedelta(seconds=self.confirmation_seconds)
            confirm_ticks = ticks_df[ticks_df['timestamp'] >= cutoff_time]
            
            if len(confirm_ticks) == 0:
                logger.debug(f"No ticks found for confirmation period ({self.confirmation_seconds}s)")
                return False
                
            confirmation_price = float(confirm_ticks['price'].iloc[-1])
            
            # Log key metrics
            logger.info(
                f"Current price: {current_price:.4f}, "
                f"{self.lookback_seconds}s ago: {oldest_price:.4f}, "
                f"Change: {pct_change:.4f}%, "
                f"Confirmation price: {confirmation_price:.4f}"
            )
            
            # Check volume if required
            volume_condition = True
            if self.use_volume and 'volume' in ticks_df.columns:
                # Implement volume condition here
                pass
            
            # Check if minute-based conditions are met
            minute_condition = self.check_minute_conditions()
            
            # Check conditions based on direction
            if self.direction == 'LONG':
                # For long entries:
                # 1. Current price must be higher than lookback price by threshold
                # 2. Current price must be at least as high as confirmation price
                # 3. Minute conditions must be met if enabled
                return (pct_change >= self.momentum_threshold and 
                        current_price >= confirmation_price and
                        volume_condition and
                        minute_condition)
            else:  # SHORT
                # For short entries:
                # 1. Current price must be lower than lookback price by threshold
                # 2. Current price must be at most as high as confirmation price
                # 3. Minute conditions must be met if enabled
                return (pct_change <= -abs(self.momentum_threshold) and 
                        current_price <= confirmation_price and
                        volume_condition and
                        minute_condition)
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """
        Check if exit conditions are met based on current price and position data.
        Uses a trailing stop based on the extreme price since entry.
        
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
