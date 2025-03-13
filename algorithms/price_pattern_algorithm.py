
"""
Price Pattern Algorithm

This algorithm detects common chart patterns like double tops/bottoms
and head-and-shoulders formations at the tick level.
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class Price_patternAlgorithm:
    def __init__(self, direction, parameters):
        """Initialize with direction and parameters from YAML config"""
        self.direction = direction
        
        # Extract parameters with defaults
        self.lookback_ticks = parameters.get('lookback_ticks', 300)
        self.smoothing_period = parameters.get('smoothing_period', 5)  # For noise reduction
        self.peak_height = parameters.get('peak_height', 0.15)  # Min height as % of range
        self.peak_distance = parameters.get('peak_distance', 20)  # Min distance between peaks
        self.pattern_symmetry = parameters.get('pattern_symmetry', 0.7)  # 1.0 = perfectly symmetric
        self.neckline_break_pct = parameters.get('neckline_break_pct', 0.1)  # Breakout confirmation %
        self.target_factor = parameters.get('target_factor', 1.0)  # Target as multiple of pattern height
        
        logger.info(f"Initialized {direction} price pattern algorithm with "
                   f"lookback={self.lookback_ticks}, "
                   f"smoothing={self.smoothing_period}, "
                   f"peak_height={self.peak_height}%")
        
    def _smooth_prices(self, prices):
        """Apply smoothing to reduce noise"""
        return pd.Series(prices).rolling(self.smoothing_period, min_periods=1).mean().values
        
    def _find_peaks_troughs(self, prices):
        """Find significant peaks and troughs in price data"""
        smoothed = self._smooth_prices(prices)
        price_range = max(smoothed) - min(smoothed)
        min_height = price_range * (self.peak_height / 100)
        
        # Find peaks (high points)
        peaks, _ = find_peaks(smoothed, height=min_height, distance=self.peak_distance)
        
        # Find troughs (low points) by inverting the data
        troughs, _ = find_peaks(-smoothed, height=min_height, distance=self.peak_distance)
        
        peak_values = [(i, smoothed[i]) for i in peaks]
        trough_values = [(i, smoothed[i]) for i in troughs]
        
        return peak_values, trough_values, smoothed
    
    def _detect_double_top(self, peaks, troughs, smoothed):
        """Detect double top pattern (bearish)"""
        if len(peaks) < 2 or len(troughs) < 1:
            return None
            
        # Get the last two peaks and the trough between them
        last_peaks = sorted(peaks[-2:], key=lambda x: x[0])
        
        # Find troughs between these peaks
        middle_troughs = [t for t in troughs if last_peaks[0][0] < t[0] < last_peaks[1][0]]
        if not middle_troughs:
            return None
            
        middle_trough = sorted(middle_troughs, key=lambda x: x[1])[0]  # Lowest trough
        
        # Check if peaks are at similar heights (symmetry)
        peak_height_diff = abs(last_peaks[0][1] - last_peaks[1][1])
        avg_peak_height = (last_peaks[0][1] + last_peaks[1][1]) / 2
        pattern_height = avg_peak_height - middle_trough[1]
        
        # Check symmetry and valid pattern height
        if (peak_height_diff / avg_peak_height <= (1 - self.pattern_symmetry) and
            pattern_height > 0):
            
            # Calculate neckline (support level at the trough)
            neckline = middle_trough[1]
            
            # Current price should be below the neckline to confirm the pattern
            current_price = smoothed[0]
            
            # Check for confirmation
            if current_price < neckline - (pattern_height * self.neckline_break_pct / 100):
                return {
                    'pattern': 'double_top',
                    'neckline': neckline,
                    'pattern_height': pattern_height,
                    'target': neckline - pattern_height * self.target_factor
                }
        
        return None
        
    def _detect_double_bottom(self, peaks, troughs, smoothed):
        """Detect double bottom pattern (bullish)"""
        if len(troughs) < 2 or len(peaks) < 1:
            return None
            
        # Get the last two troughs and the peak between them
        last_troughs = sorted(troughs[-2:], key=lambda x: x[0])
        
        # Find peaks between these troughs
        middle_peaks = [p for p in peaks if last_troughs[0][0] < p[0] < last_troughs[1][0]]
        if not middle_peaks:
            return None
            
        middle_peak = sorted(middle_peaks, key=lambda x: x[1], reverse=True)[0]  # Highest peak
        
        # Check if troughs are at similar heights (symmetry)
        trough_height_diff = abs(last_troughs[0][1] - last_troughs[1][1])
        avg_trough_height = (last_troughs[0][1] + last_troughs[1][1]) / 2
        pattern_height = middle_peak[1] - avg_trough_height
        
        # Check symmetry and valid pattern height
        if (trough_height_diff / avg_trough_height <= (1 - self.pattern_symmetry) and
            pattern_height > 0):
            
            # Calculate neckline (resistance level at the peak)
            neckline = middle_peak[1]
            
            # Current price should be above the neckline to confirm the pattern
            current_price = smoothed[0]
            
            # Check for confirmation
            if current_price > neckline + (pattern_height * self.neckline_break_pct / 100):
                return {
                    'pattern': 'double_bottom',
                    'neckline': neckline,
                    'pattern_height': pattern_height,
                    'target': neckline + pattern_height * self.target_factor
                }
        
        return None
        
    def check_entry_conditions(self, ticks_df):
        """Check if entry conditions are met based on tick data"""
        if ticks_df is None or len(ticks_df) < self.lookback_ticks:
            return False
            
        try:
            # Extract prices and reverse order (index 0 = oldest)
            prices = ticks_df['price'].values[::-1]
            current_price = prices[-1]  # Most recent price
            
            # Find peaks and troughs
            peaks, troughs, smoothed = self._find_peaks_troughs(prices)
            
            # Check pattern based on direction
            pattern = None
            if self.direction == 'LONG':
                # Look for bullish patterns (double bottom)
                pattern = self._detect_double_bottom(peaks, troughs, smoothed[::-1])  # Flip back for current price check
                if pattern:
                    logger.info(f"LONG double bottom detected. "
                               f"Current: {current_price:.4f}, "
                               f"Neckline: {pattern['neckline']:.4f}, "
                               f"Target: {pattern['target']:.4f}")
            else:  # SHORT
                # Look for bearish patterns (double top)
                pattern = self._detect_double_top(peaks, troughs, smoothed[::-1])  # Flip back for current price check
                if pattern:
                    logger.info(f"SHORT double top detected. "
                               f"Current: {current_price:.4f}, "
                               f"Neckline: {pattern['neckline']:.4f}, "
                               f"Target: {pattern['target']:.4f}")
            
            # If pattern detected, store in position_data for exit conditions
            if pattern:
                # We'd store this in the bot's position_data when opening a position
                # For now, return True to indicate entry conditions are met
                return True
                
            return False
                
        except Exception as e:
            logger.error(f"Error in check_entry_conditions: {e}")
            return False
    
    def check_exit_conditions(self, current_price, position_data):
        """Check if exit conditions are met"""
        try:
            entry_price = position_data.get('entry_price', current_price)
            pattern = position_data.get('pattern', {})
            
            # If no pattern data stored, use default exit logic
            if not pattern:
                if self.direction == 'LONG':
                    # Default long exit: 1.5% profit or 1% loss
                    return current_price >= entry_price * 1.015 or current_price <= entry_price * 0.99
                else:
                    # Default short exit: 1.5% profit or 1% loss
                    return current_price <= entry_price * 0.985 or current_price >= entry_price * 1.01
            
            # Pattern-based exits
            target_price = pattern.get('target', entry_price)
            pattern_height = pattern.get('pattern_height', 0)
            neckline = pattern.get('neckline', entry_price)
            
            if self.direction == 'LONG':
                # Target profit or stop at neckline violation
                stop_price = neckline - (pattern_height * 0.1)  # 10% below neckline
                return current_price >= target_price or current_price <= stop_price
            else:  # SHORT
                # Target profit or stop at neckline violation
                stop_price = neckline + (pattern_height * 0.1)  # 10% above neckline
                return current_price <= target_price or current_price >= stop_price
                
        except Exception as e:
            logger.error(f"Error in check_exit_conditions: {e}")
            return False
