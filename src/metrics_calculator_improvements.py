"""
Enhanced Metrics Calculator with improved calculations

This module contains improvements to the existing metrics calculation system:
1. Fixed time-based metrics (one_hour_performance, two_hour_performance)
2. Corrected implementation of profit_factor and sharpe_ratio
3. Added proper calculation of avg_trade_duration
4. Improved model score calculations
5. Added new metrics such as Sortino ratio and Calmar ratio
"""

import asyncpg
import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import math
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMetricsCalculator:
    """
    Enhanced metrics calculator with improved calculation methods
    and additional risk metrics.
    """
    
    def __init__(self, db_pool):
        """
        Initialize the metrics calculator.
        
        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool
    
    def _ensure_float(self, value, default=0.0):
        """
        Convert value to float, handling None, NaN, and infinity values safely.
        
        Args:
            value: Value to convert to float
            default: Default value to return if conversion fails
            
        Returns:
            float: Converted value or default if conversion fails
        """
        if value is None:
            return default
        
        try:
            # Convert Decimal to float
            if isinstance(value, Decimal):
                value = float(value)
            
            # Handle float conversion
            result = float(value)
            
            # Check for NaN or infinite values
            if math.isnan(result) or math.isinf(result):
                return default
            
            return result
        except (ValueError, TypeError, OverflowError):
            return default
    
    def _limit_decimal_value(self, value, precision, scale):
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
    
    async def get_all_trades(self, bot_id: int, ticker: Optional[str] = None, 
                           status: str = 'closed', days: Optional[int] = None) -> List[Dict]:
        """
        Get all trades for a bot with optional filtering.
        
        Args:
            bot_id: Bot ID to fetch trades for
            ticker: Optional ticker symbol to filter by
            status: Trade status to filter by (default: 'closed')
            days: Optional number of days to look back
            
        Returns:
            List of trade dictionaries
        """
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM sim_bot_trades 
                WHERE bot_id = $1 
                AND trade_status = $2
            """
            params = [bot_id, status]
            
            if ticker:
                query += " AND ticker = $3"
                params.append(ticker)
                
                if days:
                    query += " AND entry_time >= NOW() - INTERVAL '$4 days'"
                    params.append(days)
            elif days:
                query += " AND entry_time >= NOW() - INTERVAL '$3 days'"
                params.append(days)
                
            query += " ORDER BY entry_time"
            
            return await conn.fetch(query, *params)
    
    async def calculate_one_hour_performance(self, bot_id: int, ticker: str) -> float:
        """
        Calculate performance over the last hour.
        
        Args:
            bot_id: Bot ID
            ticker: Ticker symbol
            
        Returns:
            float: Sum of PnL from trades in the last hour
        """
        async with self.db_pool.acquire() as conn:
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            result = await conn.fetchval("""
                SELECT COALESCE(SUM(trade_pnl), 0) AS performance
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND ticker = $2
                AND trade_status = 'closed'
                AND entry_time >= $3;
            """, bot_id, ticker, one_hour_ago)
            
            return self._ensure_float(result)
    
    async def calculate_two_hour_performance(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate performance over the last two hours.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Sum of PnL from trades in the last two hours
        """
        async with self.db_pool.acquire() as conn:
            two_hours_ago = datetime.now() - timedelta(hours=2)
            
            result = await conn.fetchval("""
                SELECT COALESCE(SUM(trade_pnl), 0) AS two_hour_performance
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND trade_status = 'closed'
                AND entry_time >= $2;
            """, bot_id, two_hours_ago)
            
            return self._ensure_float(result)
    
    async def calculate_avg_win_rate(self, bot_id: int, ticker: str) -> float:
        """
        Calculate average win rate.
        
        Args:
            bot_id: Bot ID
            ticker: Ticker symbol
            
        Returns:
            float: Win rate as a percentage
        """
        try:
            async with self.db_pool.acquire() as connection:
                query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM sim_bot_trades
                    WHERE bot_id = $1 
                    AND ticker = $2
                    AND trade_status = 'closed'
                """
                
                row = await connection.fetchrow(query, bot_id, ticker)
                
                if not row:
                    return 0.0
                    
                # Convert to float before division
                total_trades = self._ensure_float(row['total_trades'])
                winning_trades = self._ensure_float(row['winning_trades'])
                
                if total_trades == 0:
                    return 0.0
                    
                win_rate = (winning_trades / total_trades) * 100
                # Ensure win rate is within DECIMAL(6,2) limits
                limited_win_rate = self._limit_decimal_value(win_rate, 6, 2)
                return limited_win_rate
        except Exception as e:
            logging.error(f"Error calculating average win rate for bot {bot_id}, ticker {ticker}: {e}")
            return 0.0
    
    async def calculate_total_pnl(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate total profit/loss.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Total PnL
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COALESCE(SUM(trade_pnl), 0) AS total_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 
                AND trade_status = 'closed';
            """, bot_id)
            
            return self._ensure_float(result)
    
    async def calculate_profit_factor(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Profit factor
        """
        async with self.db_pool.acquire() as conn:
            # Get gross profit and gross loss separately to avoid division issues
            gross_profit = await conn.fetchval("""
                SELECT COALESCE(SUM(trade_pnl), 0)
                FROM sim_bot_trades
                WHERE bot_id = $1
                AND trade_status = 'closed'
                AND trade_pnl > 0;
            """, bot_id)
            
            gross_loss = await conn.fetchval("""
                SELECT COALESCE(ABS(SUM(trade_pnl)), 0)
                FROM sim_bot_trades
                WHERE bot_id = $1
                AND trade_status = 'closed'
                AND trade_pnl < 0;
            """, bot_id)
            
            gross_profit = self._ensure_float(gross_profit)
            gross_loss = self._ensure_float(gross_loss)
            
            # Calculate profit factor with safety for zero losses
            if gross_loss == 0:
                return 1.0 if gross_profit == 0 else 99.0  # Return 99 as cap for perfect performance
                
            return gross_profit / gross_loss
    
    async def calculate_sharpe_ratio(self, bot_id: int, algo_id: int, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe Ratio with improved implementation.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            risk_free_rate: Risk-free rate (default: 0.02 or 2%)
            
        Returns:
            float: Sharpe ratio
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Get trades grouped by day
                daily_returns = await connection.fetch("""
                    SELECT 
                        DATE(entry_time) as trade_date,
                        SUM(trade_pnl) as daily_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    GROUP BY DATE(entry_time)
                    ORDER BY trade_date;
                """, bot_id)
                
                if not daily_returns:
                    return 0.0
                
                # Calculate average return and standard deviation
                returns = [float(row['daily_pnl']) for row in daily_returns]
                
                if len(returns) < 2:
                    return 0.0  # Not enough data
                
                # Calculate mean and standard deviation using numpy for accuracy
                mean_return = np.mean(returns)
                std_dev = np.std(returns, ddof=1)  # Using ddof=1 for sample std dev
                
                # Handle zero standard deviation
                if std_dev == 0:
                    return 0.0
                
                # Get annualized metrics (assuming daily returns)
                trading_days_per_year = 252
                annualized_return = mean_return * trading_days_per_year
                annualized_std_dev = std_dev * math.sqrt(trading_days_per_year)
                
                # Calculate Sharpe ratio
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std_dev
                
                return round(float(sharpe_ratio), 4)
                
        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_sortino_ratio(self, bot_id: int, algo_id: int, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sortino Ratio (like Sharpe but only considers downside risk).
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            risk_free_rate: Risk-free rate (default: 0.02 or 2%)
            
        Returns:
            float: Sortino ratio
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Get trades grouped by day
                daily_returns = await connection.fetch("""
                    SELECT 
                        DATE(entry_time) as trade_date,
                        SUM(trade_pnl) as daily_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    GROUP BY DATE(entry_time)
                    ORDER BY trade_date;
                """, bot_id)
                
                if not daily_returns:
                    return 0.0
                
                # Calculate average return and downside deviation
                returns = [float(row['daily_pnl']) for row in daily_returns]
                
                if len(returns) < 2:
                    return 0.0  # Not enough data
                
                # Calculate mean return
                mean_return = np.mean(returns)
                
                # Calculate downside deviation (only negative returns)
                # Downside returns are returns below zero (or below target, typically zero)
                downside_returns = [r for r in returns if r < 0]
                
                if not downside_returns:
                    return 99.0  # No downside returns, cap at 99
                
                downside_deviation = np.std(downside_returns, ddof=1)
                
                if downside_deviation == 0:
                    return 0.0
                
                # Get annualized metrics (assuming daily returns)
                trading_days_per_year = 252
                annualized_return = mean_return * trading_days_per_year
                annualized_downside_dev = downside_deviation * math.sqrt(trading_days_per_year)
                
                # Calculate Sortino ratio
                sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_dev
                
                return round(float(sortino_ratio), 4)
                
        except Exception as e:
            logging.error(f"Error calculating Sortino ratio for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_avg_trade_duration(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate average trade duration in seconds.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Average trade duration in seconds
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Get all completed trades
                trades = await connection.fetch("""
                    SELECT 
                        entry_time, exit_time
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    AND exit_time IS NOT NULL
                """, bot_id)
                
                if not trades:
                    return 0.0
                
                # Calculate duration for each trade
                durations = []
                for trade in trades:
                    if trade['entry_time'] and trade['exit_time']:
                        duration = (trade['exit_time'] - trade['entry_time']).total_seconds()
                        durations.append(duration)
                
                if not durations:
                    return 0.0
                
                avg_duration = sum(durations) / len(durations)
                return round(avg_duration, 2)
                
        except Exception as e:
            logging.error(f"Error calculating average trade duration for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_drawdowns(self, bot_id: int, algo_id: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate drawdowns with enhanced metrics.
        
        Args:
            bot_id: Bot ID
            algo_id: Optional algorithm ID
            
        Returns:
            dict: Dictionary with drawdown metrics
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Get all trades sorted by time
                trades = await connection.fetch("""
                    SELECT 
                        trade_id, trade_pnl, entry_time, exit_time
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    ORDER BY entry_time
                """, bot_id)
                
                if not trades:
                    return {
                        "avg_drawdown": 0.0, 
                        "max_drawdown": 0.0,
                        "max_drawdown_duration": 0.0,
                        "recovery_factor": 0.0,
                        "time_in_drawdown": 0.0,
                        "drawdown_percent": 0.0
                    }
                
                # Calculate running PnL and track drawdowns
                running_pnl = Decimal('0.0') # Use Decimal for consistency
                peak_pnl = Decimal('0.0')    # Use Decimal
                drawdowns = []
                current_drawdown_start = None
                drawdown_periods = []
                total_time_in_drawdown = Decimal('0.0') # Use Decimal
                
                for i, trade in enumerate(trades):
                    # Ensure trade_pnl is Decimal
                    if trade['trade_pnl'] is not None:
                        pnl = Decimal(str(trade['trade_pnl'])) # Convert DB Decimal/float to Decimal
                    else:
                        pnl = Decimal('0.0')
                        
                    running_pnl += pnl
                    
                    if running_pnl > peak_pnl:
                        # New equity peak
                        peak_pnl = running_pnl
                        
                        # If we were in a drawdown, record its duration
                        if current_drawdown_start is not None:
                            drawdown_end = trade['entry_time'] # entry_time should be datetime
                            if drawdown_end and current_drawdown_start:
                                drawdown_duration_td = drawdown_end - current_drawdown_start
                                drawdown_duration = Decimal(str(drawdown_duration_td.total_seconds()))
                                drawdown_periods.append(drawdown_duration)
                                total_time_in_drawdown += drawdown_duration
                            current_drawdown_start = None
                    else:
                        # In drawdown
                        current_drawdown = peak_pnl - running_pnl # Decimal calculation
                        drawdowns.append(current_drawdown)
                        
                        # Track drawdown start time
                        if current_drawdown_start is None:
                            current_drawdown_start = trade['entry_time'] # entry_time should be datetime
                
                # Calculate drawdown metrics
                if drawdowns:
                    # Ensure all calculations use Decimal
                    avg_drawdown = sum(drawdowns) / Decimal(len(drawdowns))
                    max_drawdown = max(drawdowns)
                    
                    # Calculate max drawdown duration
                    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else Decimal('0.0')
                    
                    # Calculate recovery factor (total return / max drawdown)
                    total_return = Decimal(str(trades[-1]['trade_pnl'])) if trades and trades[-1]['trade_pnl'] is not None else Decimal('0.0')
                    recovery_factor = abs(total_return / max_drawdown) if max_drawdown > Decimal('0.0') else Decimal('0.0')
                    
                    # Calculate drawdown as percentage of peak equity
                    drawdown_percent = (max_drawdown / peak_pnl * Decimal('100.0')) if peak_pnl > Decimal('0.0') else Decimal('0.0')
                else:
                    avg_drawdown = Decimal('0.0')
                    max_drawdown = Decimal('0.0')
                    max_drawdown_duration = Decimal('0.0')
                    recovery_factor = Decimal('0.0')
                    drawdown_percent = Decimal('0.0')
                
                # Return results as floats for compatibility, using _ensure_float
                return {
                    "avg_drawdown": self._ensure_float(avg_drawdown),
                    "max_drawdown": self._ensure_float(max_drawdown),
                    "max_drawdown_duration": self._ensure_float(max_drawdown_duration),
                    "recovery_factor": self._ensure_float(recovery_factor),
                    "time_in_drawdown": self._ensure_float(total_time_in_drawdown), # Time stored as seconds
                    "drawdown_percent": self._ensure_float(drawdown_percent)
                }
                
        except Exception as e:
            logging.error(f"Error calculating drawdowns for bot {bot_id}: {e}")
            return {
                "avg_drawdown": 0.0, 
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0.0,
                "recovery_factor": 0.0,
                "time_in_drawdown": 0.0,
                "drawdown_percent": 0.0
            }
    
    async def calculate_calmar_ratio(self, bot_id: int, algo_id: Optional[int] = None) -> float:
        """
        Calculate the Calmar Ratio (annualized return / maximum drawdown).
        
        Args:
            bot_id: Bot ID
            algo_id: Optional algorithm ID
            
        Returns:
            float: Calmar ratio
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Calculate annualized return
                daily_returns = await connection.fetch("""
                    SELECT 
                        DATE(entry_time) as trade_date,
                        SUM(trade_pnl) as daily_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    GROUP BY DATE(entry_time)
                    ORDER BY trade_date;
                """, bot_id)
                
                if not daily_returns or len(daily_returns) < 2:
                    return 0.0
                
                # Calculate average daily return
                returns = [float(row['daily_pnl']) for row in daily_returns]
                avg_daily_return = np.mean(returns)
                
                # Annualize the return (assuming 252 trading days per year)
                annualized_return = avg_daily_return * 252
                
                # Get maximum drawdown
                drawdown_info = await self.calculate_drawdowns(bot_id, algo_id)
                max_drawdown = drawdown_info['max_drawdown']
                
                if max_drawdown == 0:
                    return 0.0  # Avoid division by zero
                
                # Calculate Calmar ratio
                calmar_ratio = annualized_return / max_drawdown
                
                return round(float(calmar_ratio), 4)
                
        except Exception as e:
            logging.error(f"Error calculating Calmar ratio for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_r_multiple(self, bot_id: int, algo_id: Optional[int] = None) -> float:
        """
        Calculate R-Multiple (average win / average loss).
        
        Args:
            bot_id: Bot ID
            algo_id: Optional algorithm ID
            
        Returns:
            float: R-Multiple
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Calculate average win
                avg_win = await connection.fetchval("""
                    SELECT AVG(trade_pnl)
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    AND trade_pnl > 0
                """, bot_id)
                
                # Calculate average loss (absolute value)
                avg_loss = await connection.fetchval("""
                    SELECT ABS(AVG(trade_pnl))
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    AND trade_pnl < 0
                """, bot_id)
                
                avg_win = self._ensure_float(avg_win)
                avg_loss = self._ensure_float(avg_loss)
                
                if avg_loss == 0:
                    return 99.0 if avg_win > 0 else 0.0  # Cap at 99 for perfect performance
                
                r_multiple = avg_win / avg_loss
                
                return round(float(r_multiple), 2)
                
        except Exception as e:
            logging.error(f"Error calculating R-Multiple for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_all_metrics(self, bot_id: int, ticker: str) -> Dict[str, Any]:
        """
        Calculate all metrics for a bot in a single function.
        
        Args:
            bot_id: Bot ID
            ticker: Ticker symbol
            
        Returns:
            dict: Dictionary with all calculated metrics
        """
        try:
            # Run all metric calculations concurrently
            one_hour_perf, two_hour_perf, avg_win_rate, total_pnl, avg_profit_per_trade, \
            profit_factor, drawdown_info, sharpe_ratio, sortino_ratio, calmar_ratio, \
            r_multiple, avg_trade_duration = await asyncio.gather(
                self.calculate_one_hour_performance(bot_id, ticker),
                self.calculate_two_hour_performance(bot_id, bot_id),  # Using bot_id as algo_id
                self.calculate_avg_win_rate(bot_id, ticker),
                self.calculate_total_pnl(bot_id, bot_id),
                self.calculate_avg_profit_per_trade(bot_id, bot_id),
                self.calculate_profit_factor(bot_id, bot_id),
                self.calculate_drawdowns(bot_id, bot_id),
                self.calculate_sharpe_ratio(bot_id, bot_id),
                self.calculate_sortino_ratio(bot_id, bot_id),
                self.calculate_calmar_ratio(bot_id, bot_id),
                self.calculate_r_multiple(bot_id, bot_id),
                self.calculate_avg_trade_duration(bot_id, bot_id)
            )
            
            # Additional time-frame metrics
            one_day_perf = await self.calculate_performance_over_period(bot_id, bot_id, timedelta(days=1))
            one_week_perf = await self.calculate_performance_over_period(bot_id, bot_id, timedelta(weeks=1))
            one_month_perf = await self.calculate_performance_over_period(bot_id, bot_id, timedelta(days=30))
            
            # Combined metrics dictionary
            return {
                # Time-based performance
                "one_hour_performance": one_hour_perf,
                "two_hour_performance": two_hour_perf,
                "one_day_performance": one_day_perf,
                "one_week_performance": one_week_perf,
                "one_month_performance": one_month_perf,
                
                # Win rates and profitability
                "avg_win_rate": avg_win_rate,
                "total_pnl": total_pnl,
                "avg_profit_per_trade": avg_profit_per_trade,
                "profit_factor": profit_factor,
                "r_multiple": r_multiple,
                
                # Drawdowns
                "avg_drawdown": drawdown_info["avg_drawdown"],
                "max_drawdown": drawdown_info["max_drawdown"],
                "max_drawdown_duration": drawdown_info["max_drawdown_duration"],
                "recovery_factor": drawdown_info["recovery_factor"],
                "time_in_drawdown": drawdown_info["time_in_drawdown"],
                "drawdown_percent": drawdown_info["drawdown_percent"],
                
                # Risk metrics
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                
                # Trade metrics
                "avg_trade_duration": avg_trade_duration
            }
            
        except Exception as e:
            logging.error(f"Error calculating all metrics for bot {bot_id}, ticker {ticker}: {e}")
            return {}
    
    # Include methods from the original calculator for compatibility
    
    async def calculate_avg_profit_per_trade(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate average profit per trade.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Average profit per trade
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COALESCE(AVG(trade_pnl), 0) AS avg_profit_per_trade
                FROM sim_bot_trades
                WHERE bot_id = $1
                AND trade_status = 'closed';
            """, bot_id)
            
            return self._ensure_float(result)
    
    async def calculate_total_trades(self, bot_id: int, algo_id: int) -> int:
        """
        Calculate total number of trades.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            int: Total number of trades
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) AS total_trades
                FROM sim_bot_trades
                WHERE bot_id = $1
                AND trade_status = 'closed';
            """, bot_id)
            
            return result
    
    async def calculate_performance_over_period(self, bot_id: int, algo_id: int, period: timedelta) -> float:
        """
        Calculate performance over a specific time period.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            period: Time period
            
        Returns:
            float: Performance over the period
        """
        async with self.db_pool.acquire() as connection:
            end_time = datetime.now()
            start_time = end_time - period
            result = await connection.fetchval("""
                SELECT COALESCE(SUM(trade_pnl), 0)
                FROM sim_bot_trades
                WHERE bot_id = $1
                AND trade_status = 'closed' 
                AND entry_time BETWEEN $2 AND $3;
            """, bot_id, start_time, end_time)
            
            # Return formatted value if result is not None, else return 0
            return round(self._ensure_float(result), 2)
    
    async def calculate_one_day_performance(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate performance over the last day.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Performance over the last day
        """
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(days=1))
    
    async def calculate_one_week_performance(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate performance over the last week.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Performance over the last week
        """
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(weeks=1))
    
    async def calculate_one_month_performance(self, bot_id: int, algo_id: int) -> float:
        """
        Calculate performance over the last month.
        
        Args:
            bot_id: Bot ID
            algo_id: Algorithm ID
            
        Returns:
            float: Performance over the last month
        """
        # Assuming a month of 30 days
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(days=30))
    
    async def calculate_profit_per_second(self, bot_id: int, algo_id: Optional[int] = None) -> float:
        """
        Calculate profit per second of activity.
        
        Args:
            bot_id: Bot ID
            algo_id: Optional algorithm ID
            
        Returns:
            float: Profit per second
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Query to get the total PnL and time span
                query = """
                    SELECT 
                        COALESCE(SUM(trade_pnl), 0) as total_pnl,
                        MAX(exit_time) - MIN(entry_time) as time_span
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                """
                
                row = await connection.fetchrow(query, bot_id)
                
                if not row or not row['time_span']:
                    return 0.0
                    
                # Convert total_pnl to float
                total_pnl = self._ensure_float(row['total_pnl'])
                
                # Calculate total seconds
                total_seconds = row['time_span'].total_seconds()
                
                if total_seconds == 0:
                    return 0.0
                    
                return total_pnl / total_seconds
        except Exception as e:
            logging.error(f"Error calculating profit per second for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_and_insert_win_streaks(self, bot_id: int, algo_id: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate and insert win streak statistics.
        
        Args:
            bot_id: Bot ID
            algo_id: Optional algorithm ID
            
        Returns:
            dict: Dictionary with win streak metrics
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Get all trades sorted by time
                query = """
                    SELECT trade_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    AND trade_status = 'closed'
                    ORDER BY entry_time
                """
                
                rows = await connection.fetch(query, bot_id)
                
                if not rows:
                    return {
                        "win_streak_2": 0.0,
                        "win_streak_3": 0.0,
                        "win_streak_4": 0.0,
                        "win_streak_5": 0.0,
                    }
                
                # Analyze win streaks
                streaks = {2: 0, 3: 0, 4: 0, 5: 0}
                current_streak = 0
                
                for row in rows:
                    # Ensure we have float value for trade_pnl
                    pnl = self._ensure_float(row['trade_pnl'])
                    
                    if pnl > 0:
                        current_streak += 1
                        
                        # Check if this extends a streak of interest
                        for streak_len in [2, 3, 4, 5]:
                            if current_streak >= streak_len:
                                streaks[streak_len] += 1
                    else:
                        current_streak = 0
                
                # Calculate percentages
                total_trades = len(rows)
                win_streak_metrics = {}
                
                for streak_len in [2, 3, 4, 5]:
                    # Safe division with float conversion
                    total_trades_float = self._ensure_float(total_trades)
                    if total_trades_float > 0:
                        win_streak_metrics[f"win_streak_{streak_len}"] = (streaks[streak_len] / total_trades_float) * 100
                    else:
                        win_streak_metrics[f"win_streak_{streak_len}"] = 0.0
                
                return win_streak_metrics
                
        except Exception as e:
            logging.error(f"Error calculating win streaks for bot {bot_id}: {e}")
            return {
                "win_streak_2": 0.0,
                "win_streak_3": 0.0,
                "win_streak_4": 0.0,
                "win_streak_5": 0.0,
            }