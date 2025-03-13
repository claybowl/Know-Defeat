"""
Volatility Breakout Algorithm

This algorithm adapts to the current market volatility level
and trades significant breakouts from consolidation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)

class Volatility_breakoutAlgorithm:
    def __init__(self, direction, parameters):
        """Initialize with direction and parameters from YAML config"""
        self.direction = direction
        
        # Extract parameters with defaults
        self.atr_lookback = parameters.get('atr_lookback', 100)  # Ticks for ATR calculation
        self.volatility_factor = float(parameters.get('volatility_factor', 1.5))  # Breakout multiplier
        self.consolidation_threshold = float(parameters.get('consolidation_threshold', 0.5))  # Max volatility for consolidation
        self.min_consolidation_ticks = parameters.get('min_consolidation_ticks', 30)  # Min consolidation period
        self.profit_factor = float(parameters.get('profit_factor', 2.0))  # Target as multiple of ATR
        self.stop_factor = float(parameters.get('stop_factor', 1.0))  # Stop as multiple of ATR
        
        logger.info(f"Initialized {direction} volatility breakout algorithm with "
                   f"lookback={self.atr_lookback}, "
                   f"vol_factor={self.volatility_factor}, "
                   f"profit_factor={self.profit_factor}")
        
    def _calculate_atr(self, prices):
        """Calculate Average True Range for volatility measurement"""
        # Convert all prices to float to avoid Decimal/float type issues
        prices = [float(p) for p in prices]
        price_changes = np.abs(np.diff(prices))
        if len(price_changes) == 0:
            return 0
        return np.mean(price_changes)
        
    def _is_consolidating(self, prices, atr):
        """Check if market is in a consolidation phase"""
        # Use a shorter lookback for consolidation check
        recent_prices = prices[:self.min_consolidation_ticks]
        
        if len(recent_prices) < self.min_consolidation_ticks:
            return False
            
        # Calculate recent price range - convert to float
        price_range = float(max(recent_prices)) - float(min(recent_prices))
        
        # Calculate average price for percentage comparison
        # Convert to list of floats first
        recent_prices_float = [float(p) for p in recent_prices]
        avg_price = np.mean(recent_prices_float)
        
        # Check if range is small relative to volatility
        return price_range <= (float(atr) * self.consolidation_threshold * self.min_consolidation_ticks)
        
    def check_entry_conditions(self, ticks_df):
        """Check if entry conditions are met based on tick data"""
        if ticks_df is None or len(ticks_df) < self.atr_lookback:
            return False
            
        try:
            # Extract prices and ensure they are floats
            prices = ticks_df['price'].values
            # Convert current price to float to avoid Decimal/float issues
            current_price = float(prices[0])
            
            # Calculate ATR (Average True Range)
            atr = self._calculate_atr(prices)
            
            # Check for consolidation phase
            if not self._is_consolidating(prices, atr):
                logger.debug("Market not in consolidation phase")
                return False
                
            # Define breakout level based on consolidation
            consolidation_prices = prices[:self.min_consolidation_ticks]
            consolidation_high = float(max(consolidation_prices))
            consolidation_low = float(min(consolidation_prices))
            
            # Calculate breakout thresholds - ensure all values are floats
            atr_float = float(atr)
            breakout_up = consolidation_high + (atr_float * self.volatility_factor)
            breakout_down = consolidation_low - (atr_float * self.volatility_factor)
            
            logger.info(f"ATR: {atr_float:.4f}, Consolidation: {consolidation_low:.4f}-{consolidation_high:.4f}, "
                       f"Breakouts: {breakout_down:.4f}/{breakout_up:.4f}, Current: {current_price:.4f}")
            
            # Check breakout conditions based on direction
            if self.direction == 'LONG':
                # For LONG, check if price breaks above consolidation
                breakout = current_price > breakout_up
                if breakout:
                    logger.info(f"LONG volatility breakout at {current_price:.4f} (> {breakout_up:.4f})")
                    
                    # Store ATR for position management
                    position_data = {
                        'atr': atr_float,
                        'breakout_level': breakout_up
                    }
                    
                    return True
                    
            else:  # SHORT
                # For SHORT, check if price breaks below consolidation
                breakout = current_price < breakout_down
                if breakout:
                    logger.info(f"SHORT volatility breakout at {current_price:.4f} (< {breakout_down:.4f})")
                    
                    # Store ATR for position management
                    position_data = {
                        'atr': atr_float,
                        'breakout_level': breakout_down
                    }
                    
                    return True
            
            return False
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """Check if exit conditions are met"""
        try:
            # Convert all values to float to avoid Decimal issues
            current_price = float(current_price)
            entry_price = float(position_data.get('entry_price', current_price))
            atr = float(position_data.get('atr', current_price * 0.005))  # Default to 0.5% if not stored
            breakout_level = float(position_data.get('breakout_level', entry_price))
            
            if self.direction == 'LONG':
                # For long positions
                target_price = entry_price + (atr * self.profit_factor)
                
                # Dynamic stop: Either ATR-based or breakout level, whichever is higher
                stop_price = max(
                    entry_price - (atr * self.stop_factor),
                    breakout_level - (atr * 0.5)  # Half ATR below breakout level
                )
                
                # Check exit conditions
                hit_target = current_price >= target_price
                hit_stop = current_price <= stop_price
                
                if hit_target:
                    logger.info(f"LONG target hit: {current_price:.4f} >= {target_price:.4f}")
                if hit_stop:
                    logger.info(f"LONG stop hit: {current_price:.4f} <= {stop_price:.4f}")
                    
                return hit_target or hit_stop
                
            else:  # SHORT
                # For short positions
                target_price = entry_price - (atr * self.profit_factor)
                
                # Dynamic stop: Either ATR-based or breakout level, whichever is lower
                stop_price = min(
                    entry_price + (atr * self.stop_factor),
                    breakout_level + (atr * 0.5)  # Half ATR above breakout level
                )
                
                # Check exit conditions
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
