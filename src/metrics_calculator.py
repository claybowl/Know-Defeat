import asyncpg
import logging
import asyncpg
from datetime import datetime, timedelta
from decimal import Decimal
import math

class MetricsCalculator:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    def _ensure_float(self, value, default=0.0):
        """Convert value to float, handling None, NaN, and infinity values safely."""
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

    async def calculate_one_hour_performance(self, bot_id, ticker):
        async with self.db_pool.acquire() as connection:
            result = await connection.fetchval("""
                SELECT AVG(trade_pnl) AS performance
                FROM sim_bot_trades
                WHERE bot_id = $1 AND ticker = $2
                AND entry_time >= NOW() - INTERVAL '1 hour';
            """, bot_id, ticker)
            return self._ensure_float(result) or 0.0

    async def calculate_avg_win_rate(self, bot_id, ticker):
        try:
            async with self.db_pool.acquire() as connection:
                query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM sim_bot_trades
                    WHERE bot_id = $1 AND ticker = $2
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

    async def calculate_total_pnl(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT SUM(trade_pnl) AS total_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return self._ensure_float(result)

    async def calculate_avg_profit_per_trade(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT AVG(trade_pnl) AS avg_profit_per_trade
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return self._ensure_float(result) or 0.0

    async def calculate_total_trades(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) AS total_trades
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_profit_factor(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT SUM(CASE WHEN trade_pnl > 0 THEN trade_pnl ELSE 0 END) /
                    NULLIF(ABS(SUM(CASE WHEN trade_pnl < 0 THEN trade_pnl ELSE 0 END)), 0) AS profit_factor
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return self._ensure_float(result) or 1.0

    async def calculate_two_hour_performance(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT 
                    (SUM(CASE WHEN entry_time >= NOW() - INTERVAL '2 hours' THEN trade_pnl ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN entry_time >= NOW() - INTERVAL '2 hours' THEN 1 ELSE 0 END), 0)) AS two_hour_performance
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return self._ensure_float(result) or 0.0

    async def calculate_performance_over_period(self, bot_id, algo_id, period):
        async with self.db_pool.acquire() as connection:
            end_time = datetime.now()
            start_time = end_time - period
            result = await connection.fetchval("""
                SELECT SUM(trade_pnl) 
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed' 
                    AND entry_time BETWEEN $3 AND $4;
            """, bot_id, algo_id, start_time, end_time)
            
            # Return formatted value if result is not None, else return 0
            return round(self._ensure_float(result) or 0, 2)

    async def calculate_one_day_performance(self, bot_id, algo_id):
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(days=1))

    async def calculate_one_week_performance(self, bot_id, algo_id):
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(weeks=1))

    async def calculate_one_month_performance(self, bot_id, algo_id):
        # Assuming a month of 30 days
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(days=30))

    async def calculate_profit_per_second(self, bot_id, algo_id=None):
        """Calculate profit per second of activity."""
        try:
            async with self.db_pool.acquire() as connection:
                # Query to get the total PnL and time span
                query = """
                    SELECT 
                        SUM(trade_pnl) as total_pnl,
                        MAX(exit_time) - MIN(entry_time) as time_span
                    FROM sim_bot_trades
                    WHERE bot_id = $1
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

    async def calculate_trade_frequency(self, bot_id, algo_id, period):
        async with self.db_pool.acquire() as connection:
            start_time = datetime.now() - period
            count = await connection.fetchval("""
                SELECT COUNT(*)
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed' 
                    AND entry_time >= $3;
            """, bot_id, algo_id, start_time)
            return count

    async def calculate_drawdowns(self, bot_id, algo_id=None):
        """Calculate average and maximum drawdowns."""
        try:
            async with self.db_pool.acquire() as connection:
                # Query to get all trade PnLs for calculating drawdowns
                query = """
                    SELECT trade_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                    ORDER BY entry_time
                """
                rows = await connection.fetch(query, bot_id)
                
                if not rows:
                    return {"avg_drawdown": 0.0, "max_drawdown": 0.0}
                
                # Convert all trade_pnl values to float
                pnl_values = [self._ensure_float(row['trade_pnl']) for row in rows]
                
                # Calculate running PnL and track drawdowns
                running_pnl = 0.0
                peak_pnl = 0.0
                drawdowns = []
                current_drawdown = 0.0
                
                for pnl in pnl_values:
                    running_pnl += pnl
                    
                    if running_pnl > peak_pnl:
                        peak_pnl = running_pnl
                        current_drawdown = 0.0
                    else:
                        current_drawdown = peak_pnl - running_pnl
                        drawdowns.append(current_drawdown)
                
                # Calculate average and maximum drawdowns
                if drawdowns:
                    avg_drawdown = sum(drawdowns) / len(drawdowns)
                    max_drawdown = max(drawdowns) if drawdowns else 0.0
                else:
                    avg_drawdown = 0.0
                    max_drawdown = 0.0
                    
                return {"avg_drawdown": avg_drawdown, "max_drawdown": max_drawdown}
        except Exception as e:
            logging.error(f"Error calculating drawdowns for bot {bot_id}: {e}")
            return {"avg_drawdown": 0.0, "max_drawdown": 0.0}

    async def calculate_and_store_sharpe_ratio(self, bot_id, algo_id):
        async with self.db_pool.acquire() as connection:
            # Calculate daily returns
            daily_returns = await connection.fetch("""
                SELECT DATE(entry_time) as trade_date,
                    SUM(trade_pnl) as daily_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2
                GROUP BY DATE(entry_time)
                ORDER BY trade_date;
            """, bot_id, algo_id)
            
            if not daily_returns:
                return 0
            
            # Calculate average return and standard deviation
            returns = [row['daily_pnl'] for row in daily_returns]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0
            std_dev = variance ** 0.5
            
            # Assuming risk-free rate of 0.02 (2%)
            risk_free_rate = 0.02
            sharpe_ratio = ((avg_return - risk_free_rate) / std_dev) if std_dev > 0 else 0
            
            # Store in bot_metrics
            await connection.execute("""
                INSERT INTO bot_metrics (bot_id, algo_id, sharpe_ratio, timestamp)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (bot_id, algo_id, DATE(timestamp))
                DO UPDATE SET sharpe_ratio = $3, last_updated = NOW();
            """, bot_id, algo_id, round(sharpe_ratio, 4))
            
            return round(sharpe_ratio, 4)

    async def calculate_and_store_atr(self, bot_id, algo_id, period=14):
        async with self.db_pool.acquire() as connection:
            # Get high, low, close prices for the period
            price_data = await connection.fetch("""
                SELECT 
                    DATE(entry_time) as trade_date,
                    MAX(entry_price) as high,
                    MIN(entry_price) as low,
                    MAX(CASE WHEN ROW_NUMBER() OVER (PARTITION BY DATE(entry_time) 
                        ORDER BY entry_time DESC) = 1 
                        THEN entry_price END) as close
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2
                GROUP BY DATE(entry_time)
                ORDER BY trade_date DESC
                LIMIT $3;
            """, bot_id, algo_id, period + 1)
            
            if len(price_data) < 2:
                return 0
            
            # Calculate ATR
            tr_values = []
            for i in range(len(price_data) - 1):
                high = price_data[i]['high']
                low = price_data[i]['low']
                prev_close = price_data[i + 1]['close']
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            atr = sum(tr_values) / len(tr_values) if tr_values else 0
            
            # Store in bot_metrics
            await connection.execute("""
                INSERT INTO bot_metrics (bot_id, algo_id, average_true_range, timestamp)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (bot_id, algo_id, DATE(timestamp))
                DO UPDATE SET average_true_range = $3, last_updated = NOW();
            """, bot_id, algo_id, round(atr, 4))
            
            return round(atr, 4)

    async def calculate_and_insert_execution_metrics(self, bot_id, algo_id):
        async with self.db_pool.acquire() as connection:
            # Fetch all required trade data
            trades = await connection.fetch("""
                SELECT entry_price, exit_price, entry_time, exit_time
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed'
            """, bot_id, algo_id)

            price_slippages = [abs(trade['exit_price'] - trade['entry_price']) for trade in trades]
            time_slippages = [(trade['exit_time'] - trade['entry_time']).total_seconds() for trade in trades]
            
            # Calculate execution metrics
            price_slippage = round(sum(price_slippages) / len(price_slippages), 4) if price_slippages else 0
            avg_trade_duration = timedelta(seconds=(sum(time_slippages) / len(time_slippages) if time_slippages else 0))

            await connection.execute("""
                UPDATE bot_metrics
                SET price_slippage = $3, avg_trade_duration = $4
                WHERE bot_id = $1 AND algo_id = $2
            """, bot_id, algo_id, price_slippage, avg_trade_duration)

    async def calculate_and_insert_win_streaks(self, bot_id, algo_id=None):
        """Calculate and insert win streak statistics."""
        try:
            async with self.db_pool.acquire() as connection:
                # Get all trades sorted by time
                query = """
                    SELECT trade_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1
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
                
                # Update the win streak metrics in the database
                try:
                    # Ensure all values are converted to float and limited to fit within DECIMAL(6,2)
                    for key, value in win_streak_metrics.items():
                        float_value = self._ensure_float(value)
                        # Apply limits for DECIMAL(6,2)
                        win_streak_metrics[key] = self._limit_decimal_value(float_value, 6, 2)
                    
                    await connection.execute("""
                        UPDATE bot_metrics
                        SET 
                            win_streak_2 = $1,
                            win_streak_3 = $2,
                            win_streak_4 = $3,
                            win_streak_5 = $4
                        WHERE bot_id = $5 AND timestamp = (
                            SELECT MAX(timestamp) 
                            FROM bot_metrics 
                            WHERE bot_id = $5
                        )
                    """, 
                    win_streak_metrics["win_streak_2"],
                    win_streak_metrics["win_streak_3"],
                    win_streak_metrics["win_streak_4"],
                    win_streak_metrics["win_streak_5"],
                    bot_id)
                    
                    return win_streak_metrics
                except Exception as e:
                    logging.error(f"Error updating win streak metrics for bot {bot_id}: {e}")
                    return win_streak_metrics
        except Exception as e:
            logging.error(f"Error calculating win streaks for bot {bot_id}: {e}")
            return {
                "win_streak_2": 0.0,
                "win_streak_3": 0.0,
                "win_streak_4": 0.0,
                "win_streak_5": 0.0,
            }

    # Add new methods for calculating missing metrics
    async def calculate_price_model_score(self, bot_id, algo_id):
        """Calculate the price prediction model score for a bot."""
        try:
            # Use a simple fallback implementation that doesn't rely on predicted_direction
            # Fallback: calculate based on win rate as a proxy
            one_day_perf = await self.calculate_one_day_performance(bot_id, algo_id)
            win_rate = await self.calculate_win_rate_over_period(bot_id, algo_id, period=timedelta(days=1))
            
            # Combine one-day performance and win rate for a score
            score = (win_rate * 0.7) + (min(max(one_day_perf, 0), 100) * 0.3)
            # Ensure score is between 0-100 and fits within DECIMAL(6,2)
            limited_score = self._limit_decimal_value(min(max(score, 0), 100), 6, 2)
            return limited_score
                
        except Exception as e:
            logging.error(f"Error calculating price model score: {e}")
            return 50.0  # Return a neutral score on error
    
    async def calculate_win_rate_over_period(self, bot_id, algo_id=None, start_time=None, end_time=None, period=None):
        """Calculate win rate over a specific time period.
        
        Parameters:
        - bot_id: ID of the bot
        - algo_id: ID of the algorithm
        - start_time: Start datetime (optional)
        - end_time: End datetime (optional)
        - period: timedelta object (optional) - if provided, will calculate start_time from end_time - period
        """
        try:
            # Handle period parameter if provided
            if period is not None:
                end_time = datetime.now()
                start_time = end_time - period
            
            async with self.db_pool.acquire() as connection:
                # Build the query with optional time filters
                query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM sim_bot_trades
                    WHERE bot_id = $1
                """
                
                params = [bot_id]
                
                if start_time:
                    query += " AND entry_time >= $2"
                    params.append(start_time)
                    
                    if end_time:
                        query += " AND entry_time <= $3"
                        params.append(end_time)
                elif end_time:
                    query += " AND entry_time <= $2"
                    params.append(end_time)
                
                # Execute the query
                row = await connection.fetchrow(query, *params)
                
                if not row or not row['total_trades']:
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
            logging.error(f"Error calculating win rate for bot {bot_id}: {e}")
            return 0.0
    
    async def calculate_volume_model_score(self, bot_id, algo_id):
        """Calculate the volume prediction model score for a bot."""
        try:
            # Simplified implementation that doesn't rely on volume_at_entry column
            # Use trade frequency as a proxy for volume effectiveness
            daily_trade_count = await self.calculate_trade_frequency(bot_id, algo_id, timedelta(days=1))
            profit_per_trade = await self.calculate_avg_profit_per_trade(bot_id, algo_id)
            
            # Scale the score based on trade count and profit per trade
            # More trades with positive profit = better volume model
            if daily_trade_count > 0 and profit_per_trade is not None and profit_per_trade > 0:
                score = min(daily_trade_count * profit_per_trade * 5, 100)  # Scale with some factor
            else:
                score = 50  # Neutral score if not enough data
            
            # Ensure the score fits within DECIMAL(6,2)
            limited_score = self._limit_decimal_value(score, 6, 2)
            return limited_score
                
        except Exception as e:
            logging.error(f"Error calculating volume model score: {e}")
            return 50.0  # Return a neutral score on error
    
    async def calculate_price_wall_score(self, bot_id, algo_id):
        """Calculate a score based on order book price walls."""
        try:
            # Simplified implementation that doesn't rely on support_level and resistance_level
            # Use the success rate of trades as a proxy
            profit_factor = await self.calculate_profit_factor(bot_id, algo_id) or 1
            win_rate = await self.calculate_win_rate_over_period(bot_id, algo_id, period=timedelta(days=7))
            
            # Combine profit factor and win rate for a score
            if profit_factor > 0:
                score = min((profit_factor * 10) + (win_rate * 0.5), 100)
            else:
                score = 50  # Neutral score if not enough data
                
            # Ensure the score fits within DECIMAL(6,2)
            limited_score = self._limit_decimal_value(score, 6, 2)
            return limited_score
                
        except Exception as e:
            logging.error(f"Error calculating price wall score: {e}")
            return 50.0  # Return a neutral score on error

    async def calculate_sharpe_ratio(self, bot_id, algo_id):
        """Calculate the Sharpe Ratio for the bot."""
        try:
            async with self.db_pool.acquire() as connection:
                # Calculate daily returns
                daily_returns = await connection.fetch("""
                    SELECT DATE(entry_time) as trade_date,
                        SUM(trade_pnl) as daily_pnl
                    FROM sim_bot_trades
                    WHERE bot_id = $1 AND algo_id = $2
                    GROUP BY DATE(entry_time)
                    ORDER BY trade_date;
                """, bot_id, algo_id)
                
                if not daily_returns:
                    return 0
                
                # Calculate average return and standard deviation
                returns = [row['daily_pnl'] for row in daily_returns]
                avg_return = sum(returns) / len(returns)
                variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0
                std_dev = variance ** 0.5
                
                # Assuming risk-free rate of 0.02 (2%)
                risk_free_rate = 0.02
                sharpe_ratio = ((avg_return - risk_free_rate) / std_dev) if std_dev > 0 else 0
                
                # Store the result
                await connection.execute("""
                    INSERT INTO bot_metrics (bot_id, algo_id, sharpe_ratio, timestamp)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (bot_id, algo_id, DATE(timestamp))
                    DO UPDATE SET sharpe_ratio = $3, last_updated = NOW();
                """, bot_id, algo_id, round(sharpe_ratio, 4))
                
                return round(sharpe_ratio, 4)
                
        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {e}")
            return 0
            
    async def calculate_average_true_range(self, bot_id, algo_id, period=14):
        """Calculate the Average True Range (ATR) for the bot."""
        try:
            async with self.db_pool.acquire() as connection:
                # Get price data
                price_data = await connection.fetch("""
                    SELECT 
                        DATE(entry_time) as trade_date,
                        MAX(entry_price) as high,
                        MIN(entry_price) as low,
                        MAX(CASE WHEN ROW_NUMBER() OVER (PARTITION BY DATE(entry_time) 
                            ORDER BY entry_time DESC) = 1 
                            THEN entry_price END) as close
                    FROM sim_bot_trades
                    WHERE bot_id = $1 AND algo_id = $2
                    GROUP BY DATE(entry_time)
                    ORDER BY trade_date DESC
                    LIMIT $3;
                """, bot_id, algo_id, period * 2)  # Get twice the period to handle calculations
                
                if len(price_data) < 2:
                    return 0
                
                # Calculate ATR
                tr_values = []
                for i in range(len(price_data) - 1):
                    high = price_data[i]['high']
                    low = price_data[i]['low']
                    prev_close = price_data[i + 1]['close']
                    
                    tr = max(
                        high - low,
                        abs(high - prev_close),
                        abs(low - prev_close)
                    )
                    tr_values.append(tr)
                
                atr = sum(tr_values) / len(tr_values) if tr_values else 0
                
                # Store the result
                await connection.execute("""
                    INSERT INTO bot_metrics (bot_id, algo_id, average_true_range, timestamp)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (bot_id, algo_id, DATE(timestamp))
                    DO UPDATE SET average_true_range = $3, last_updated = NOW();
                """, bot_id, algo_id, round(atr, 4))
                
                return round(atr, 4)
                
        except Exception as e:
            logging.error(f"Error calculating Average True Range: {e}")
            return 0
