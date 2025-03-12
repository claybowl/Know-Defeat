"""
Support-Resistance Algorithm

This algorithm identifies key support and resistance levels 
and trades bounces off these levels with confirmation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class Support_resistance_algorithmAlgorithm:
    def __init__(self, direction, parameters):
        """Initialize with direction and parameters from YAML config"""
        self.direction = direction
        
        # Extract parameters with defaults
        self.lookback_ticks = parameters.get('lookback_ticks', 500)
        self.level_threshold = parameters.get('level_threshold', 0.2)  # % range for level detection
        self.bounce_threshold = parameters.get('bounce_threshold', 0.15)  # % move to confirm bounce
        self.rejection_count = parameters.get('rejection_count', 3)  # Min touches to confirm level
        self.profit_factor = parameters.get('profit_factor', 1.5)  # Target as multiple of risk
        self.risk_pct = parameters.get('risk_pct', 0.4)  # Risk % beyond level
        
        logger.info(f"Initialized {direction} support/resistance algorithm with "
                   f"lookback={self.lookback_ticks}, "
                   f"level_threshold={self.level_threshold}%, "
                   f"min_rejections={self.rejection_count}")
        
    def _find_price_levels(self, prices):
        """Identify support and resistance levels from price data"""
        # Create price bins and count occurrences
        price_range = prices.max() - prices.min()
        bin_size = price_range * (self.level_threshold / 100)
        bins = np.arange(prices.min(), prices.max() + bin_size, bin_size)
        hist, bin_edges = np.histogram(prices, bins=bins)
        
        # Find bins with high frequency
        threshold = np.percentile(hist, 80)  # Top 20% of bins
        level_indices = np.where(hist >= threshold)[0]
        
        # Extract the price levels
        levels = [(bin_edges[i], bin_edges[i+1], hist[i]) for i in level_indices]
        
        # Sort by frequency (highest first)
        levels.sort(key=lambda x: x[2], reverse=True)
        
        return levels
    
    def _check_level_rejections(self, prices, timestamps, level_low, level_high):
        """Check how many times a level has been tested/rejected"""
        rejections = 0
        in_level = False
        
        # Scan through prices to count rejections
        for i, price in enumerate(prices):
            # Check if price is in the level zone
            if level_low <= price <= level_high:
                in_level = True
            # If we were in the level and now outside, count as rejection
            elif in_level:
                in_level = False
                rejections += 1
                
        return rejections
        
    def check_entry_conditions(self, ticks_df):
        """Check if entry conditions are met based on tick data"""
        if ticks_df is None or len(ticks_df) < self.lookback_ticks:
            return False
            
        try:
            prices = ticks_df['price'].values
            timestamps = ticks_df['timestamp'].values
            current_price = float(prices[0])
            
            # Find support/resistance levels
            levels = self._find_price_levels(prices)
            
            # Filter to only significant levels with enough rejections
            valid_levels = []
            for level_low, level_high, count in levels:
                rejections = self._check_level_rejections(prices, timestamps, level_low, level_high)
                if rejections >= self.rejection_count:
                    valid_levels.append((level_low, level_high, count, rejections))
            
            if not valid_levels:
                logger.debug("No valid support/resistance levels found")
                return False
                
            # Check for level interactions based on direction
            if self.direction == 'LONG':
                # For LONG, look for support levels below current price
                support_levels = [level for level in valid_levels if level[1] < current_price]
                
                if not support_levels:
                    return False
                    
                # Find closest support level
                closest_support = min(support_levels, key=lambda x: current_price - x[1])
                support_low, support_high, count, rejections = closest_support
                
                # Calculate distance to support
                dist_to_support = current_price - support_high
                price_change_pct = dist_to_support / current_price * 100
                
                # Check if we've just bounced off support
                bounce_condition = (price_change_pct <= self.bounce_threshold)
                
                if bounce_condition:
                    logger.info(f"LONG support bounce at {current_price:.4f}, "
                               f"Support: {support_low:.4f}-{support_high:.4f}, "
                               f"Rejections: {rejections}")
                    
                    # Store level info for exit calculations
                    position_data = {
                        'support_low': support_low,
                        'support_high': support_high
                    }
                    
                    return True
                    
            else:  # SHORT
                # For SHORT, look for resistance levels above current price
                resistance_levels = [level for level in valid_levels if level[0] > current_price]
                
                if not resistance_levels:
                    return False
                    
                # Find closest resistance level
                closest_resistance = min(resistance_levels, key=lambda x: x[0] - current_price)
                resist_low, resist_high, count, rejections = closest_resistance
                
                # Calculate distance to resistance
                dist_to_resistance = resist_low - current_price
                price_change_pct = dist_to_resistance / current_price * 100
                
                # Check if we've just bounced off resistance
                bounce_condition = (price_change_pct <= self.bounce_threshold)
                
                if bounce_condition:
                    logger.info(f"SHORT resistance bounce at {current_price:.4f}, "
                               f"Resistance: {resist_low:.4f}-{resist_high:.4f}, "
                               f"Rejections: {rejections}")
                    
                    # Store level info for exit calculations
                    position_data = {
                        'resist_low': resist_low,
                        'resist_high': resist_high
                    }
                    
                    return True
            
            return False
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """Check if exit conditions are met"""
        try:
            entry_price = position_data.get('entry_price', current_price)
            
            if self.direction == 'LONG':
                # Get support level data
                support_high = position_data.get('support_high', entry_price * 0.99)
                
                # Calculate stop and target
                stop_price = support_high - (entry_price * (self.risk_pct / 100))
                risk_amount = entry_price - stop_price
                target_price = entry_price + (risk_amount * self.profit_factor)
                
                # Check exit conditions
                return current_price >= target_price or current_price <= stop_price
                
            else:  # SHORT
                # Get resistance level data
                resist_low = position_data.get('resist_low', entry_price * 1.01)
                
                # Calculate stop and target
                stop_price = resist_low + (entry_price * (self.risk_pct / 100))
                risk_amount = stop_price - entry_price
                target_price = entry_price - (risk_amount * self.profit_factor)
                
                # Check exit conditions
                return current_price <= target_price or current_price >= stop_price
                
        except Exception as e:
            logger.error(f"Error in check_exit_conditions: {e}")
            return False
