import asyncpg
import logging
import asyncpg
from datetime import datetime, timedelta

class MetricsCalculator:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def calculate_one_hour_performance(self, bot_id, ticker):
        async with self.db_pool.acquire() as connection:
            result = await connection.fetchval("""
                SELECT AVG(performance)
                FROM trades
                WHERE bot_id = $1 AND ticker = $2
                AND timestamp >= NOW() - INTERVAL '1 hour';
            """, bot_id, ticker)
            return result or 0.0

    async def calculate_avg_win_rate(self, bot_id, ticker):
        async with self.db_pool.acquire() as connection:
            result = await connection.fetchval("""
                SELECT AVG(win_rate)
                FROM trades
                WHERE bot_id = $1 AND ticker = $2
                AND timestamp >= NOW() - INTERVAL '1 hour';
            """, bot_id, ticker)
            return result or 0.0

    async def calculate_total_pnl(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT SUM(trade_pnl) AS total_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_avg_profit_per_trade(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT AVG(trade_pnl) AS avg_profit_per_trade
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

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
                    ABS(SUM(CASE WHEN trade_pnl < 0 THEN trade_pnl ELSE 0 END)) AS profit_factor
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_two_hour_performance(self, bot_id, algo_id):
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT 
                    (SUM(CASE WHEN timestamp >= NOW() - INTERVAL '2 hours' THEN trade_pnl ELSE 0 END) /
                    SUM(CASE WHEN timestamp >= NOW() - INTERVAL '2 hours' THEN 1 ELSE 0 END)) AS two_hour_performance
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2;
            """, bot_id, algo_id)
        return result

    async def calculate_performance_over_period(self, bot_id, algo_id, period):
        async with self.db_pool.acquire() as connection:
            end_time = datetime.now()
            start_time = end_time - period
            result = await connection.fetchval("""
                SELECT SUM(trade_pnl) 
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed' 
                    AND timestamp BETWEEN $3 AND $4;
            """, bot_id, algo_id, start_time, end_time)
            
            # Return formatted value if result is not None, else return 0
            return round(result, 2) if result else 0

    async def calculate_one_day_performance(self, bot_id, algo_id):
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(days=1))

    async def calculate_one_week_performance(self, bot_id, algo_id):
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(weeks=1))

    async def calculate_one_month_performance(self, bot_id, algo_id):
        # Assuming a month of 30 days
        return await self.calculate_performance_over_period(bot_id, algo_id, timedelta(days=30))

    async def calculate_profit_per_second(self, bot_id, algo_id):
        async with self.db_pool.acquire() as connection:
            # Calculate total PnL and total time span
            total_pnl = await connection.fetchval("""
                SELECT SUM(trade_pnl)
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed';
            """, bot_id, algo_id)
            
            time_span_result = await connection.fetchrow("""
                SELECT MIN(timestamp) as start_time, MAX(timestamp) as end_time
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed';
            """, bot_id, algo_id)
            
            if time_span_result and total_pnl is not None:
                start_time, end_time = time_span_result['start_time'], time_span_result['end_time']
                total_seconds = (end_time - start_time).total_seconds()
                profit_per_second = total_pnl / total_seconds if total_seconds > 0 else 0
                return round(profit_per_second, 4)
            else:
                return 0

    async def calculate_trade_frequency(self, bot_id, algo_id, period):
        async with self.db_pool.acquire() as connection:
            start_time = datetime.now() - period
            count = await connection.fetchval("""
                SELECT COUNT(*)
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed' 
                    AND timestamp >= $3;
            """, bot_id, algo_id, start_time)
            return count

    async def calculate_drawdowns(self, bot_id, algo_id):
        async with self.db_pool.acquire() as connection:
            # Fetch trades and calculate drawdowns
            trades = await connection.fetch("""
                SELECT timestamp, trade_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed'
                ORDER BY timestamp;
            """, bot_id, algo_id)

            peak = 0
            drawdowns = []
            drawdown_durations = []
            current_drawdown = 0
            current_drawdown_start = None

            for trade in trades:
                peak = max(peak, trade['trade_pnl'])
                drawdown = peak - trade['trade_pnl']
                if drawdown > 0:
                    if current_drawdown_start is None:
                        current_drawdown_start = trade['timestamp']
                    current_drawdown += drawdown
                    drawdowns.append(drawdown)
                else:
                    if current_drawdown_start is not None:
                        drawdown_durations.append(trade['timestamp'] - current_drawdown_start)
                    current_drawdown = 0
                    current_drawdown_start = None

            avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0
            max_drawdown = max(drawdowns) if drawdowns else 0
            time_in_drawdown = sum(drawdown_durations, timedelta()) if drawdown_durations else timedelta()

            return {
                'avg_drawdown': round(avg_drawdown, 2),
                'max_drawdown': round(max_drawdown, 2),
                'time_in_drawdown': time_in_drawdown
            }

    async def calculate_and_store_sharpe_ratio(self, bot_id, algo_id):
        async with self.db_pool.acquire() as connection:
            # Calculate daily returns
            daily_returns = await connection.fetch("""
                SELECT DATE(timestamp) as trade_date,
                    SUM(trade_pnl) as daily_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2
                GROUP BY DATE(timestamp)
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
                    DATE(timestamp) as trade_date,
                    MAX(entry_price) as high,
                    MIN(entry_price) as low,
                    MAX(CASE WHEN ROW_NUMBER() OVER (PARTITION BY DATE(timestamp) 
                        ORDER BY timestamp DESC) = 1 
                        THEN entry_price END) as close
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2
                GROUP BY DATE(timestamp)
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

    async def calculate_and_insert_win_streaks(self, bot_id, algo_id):
        """Analyze trades to find win streaks and update the database with these metrics."""
        async with self.db_pool.acquire() as connection:
            trades = await connection.fetch("""
                SELECT trade_pnl
                FROM sim_bot_trades
                WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed'
                ORDER BY timestamp
            """, bot_id, algo_id)

            # Calculate win streaks
            streaks = {'win_streak_2': 0, 'win_streak_3': 0, 'win_streak_4': 0, 'win_streak_5': 0}
            current_streak = 0

            for trade in trades:
                if trade['trade_pnl'] > 0:
                    current_streak += 1
                    for key in streaks:
                        if current_streak >= int(key[-1]):
                            streaks[key] += 1
                else:
                    current_streak = 0

            total_trades = len(trades)
            if total_trades > 0:
                for key in streaks:
                    streaks[key] = round((streaks[key] / total_trades) * 100, 2)

            await connection.execute("""
                UPDATE bot_metrics
                SET win_streak_2 = $3, win_streak_3 = $4, win_streak_4 = $5, win_streak_5 = $6
                WHERE bot_id = $1 AND algo_id = $2
            """, bot_id, algo_id, streaks['win_streak_2'], streaks['win_streak_3'], streaks['win_streak_4'], streaks['win_streak_5'])

# CREATE TABLE bot_metrics (
#     -- Identifiers
#     -- Model Scores
#     price_model_score DECIMAL(6,2),
#     volume_model_score DECIMAL(6,2),
#     price_wall_score DECIMAL(6,2),
#     -- Final Rankings
#     current_rank DECIMAL(6,2),
#     last_updated TIMESTAMP DEFAULT NOW()
# );
