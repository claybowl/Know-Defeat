import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pytz
from decimal import Decimal
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from bots.coin_momentum_bot import CoinMomentumBot



class MockIBClient:
    """Simulates the IB client interface for backtesting purposes"""
    def __init__(self):
        self.next_valid_order_id = 1
        self.orders = []
        self.logger = logging.getLogger(__name__)

    def placeOrder(self, orderId, contract, order):
        """Records orders during backtesting simulation"""
        self.orders.append({
            'orderId': orderId,
            'symbol': contract.symbol,
            'action': order.action,
            'quantity': order.totalQuantity,
            'timestamp': datetime.now(pytz.UTC)
        })
        self.logger.info(f"Backtester Order: {order.action} {order.totalQuantity} {contract.symbol}")
        self.next_valid_order_id += 1

class MinuteMomentumBacktester:
    """Backtesting system for minute-based momentum strategy with trailing stops"""
    
    def __init__(self, db_pool, start_date, end_date):
        self.db_pool = db_pool
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        self.mock_ib = MockIBClient()
        
        # Strategy parameters
        self.TOTAL_USD_AMOUNT = 10000  # Position size in USD
        self.TRAILING_STOP_PERCENT = 0.002  # 0.2% trailing stop
        
        # Performance tracking
        self.trades = []
        self.cash = 100000  # Starting capital
        self.initial_capital = 100000
        self.equity_curve = []

    def in_trading_window(self, timestamp):
        """Check if the given timestamp is within valid trading hours (9:31 AM - 3:45 PM ET)"""
        et_time = timestamp.astimezone(pytz.timezone('US/Eastern'))
        
        # Check if it's a weekday
        if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        market_open = et_time.replace(hour=9, minute=31, second=0, microsecond=0)
        market_close = et_time.replace(hour=15, minute=45, second=0, microsecond=0)
        
        return market_open <= et_time <= market_close

    async def get_minute_data(self):
        """Fetch and prepare minute-by-minute price data from tick database"""
        try:
            async with self.db_pool.acquire() as conn:
                return await conn.fetch('''
                    WITH minute_prices AS (
                        SELECT 
                            date_trunc('minute', timestamp) as minute_time,
                            FIRST_VALUE(price) OVER (
                                PARTITION BY date_trunc('minute', timestamp) 
                                ORDER BY timestamp
                                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                            ) as open_price,
                            LAST_VALUE(price) OVER (
                                PARTITION BY date_trunc('minute', timestamp) 
                                ORDER BY timestamp
                                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                            ) as close_price,
                            MIN(price) OVER (PARTITION BY date_trunc('minute', timestamp)) as low_price,
                            MAX(price) OVER (PARTITION BY date_trunc('minute', timestamp)) as high_price,
                            SUM(volume) OVER (PARTITION BY date_trunc('minute', timestamp)) as volume
                        FROM tick_data
                        WHERE 
                            ticker = 'COIN'
                            AND timestamp BETWEEN $1 AND $2
                    )
                    SELECT DISTINCT
                        minute_time as timestamp,
                        open_price as "Open",
                        close_price as close,
                        LAG(open_price) OVER (ORDER BY minute_time) as prev_open,
                        LAG(close_price) OVER (ORDER BY minute_time) as prev_close
                    FROM minute_prices
                    ORDER BY minute_time
                ''', self.start_date, self.end_date)
        except Exception as e:
            self.logger.error(f"Error fetching minute data: {e}")
            self.logger.error(f"Error details: ", exc_info=True)
            return []

    def calculate_performance_metrics(self):
        """Calculate comprehensive trading performance metrics"""
        if not self.trades:
            return "No trades executed during the backtesting period."

        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate maximum drawdown
        peak = self.initial_capital
        max_drawdown = 0
        running_capital = self.initial_capital

        for trade in self.trades:
            running_capital += trade.get('pnl', 0)
            if running_capital > peak:
                peak = running_capital
            drawdown = (peak - running_capital) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        return f"""
=== Backtesting Results ===
Period: {self.start_date.date()} to {self.end_date.date()}

Performance Metrics:
------------------
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${running_capital:,.2f}
Total Return: {((running_capital/self.initial_capital - 1) * 100):.2f}%
Total P&L: ${total_pnl:,.2f}

Trade Statistics:
---------------
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {total_trades - winning_trades}
Win Rate: {win_rate:.2f}%
Average Trade P&L: ${(total_pnl/total_trades):,.2f}
Maximum Drawdown: {max_drawdown:.2f}%

Last 5 Trades:
------------
{self._format_recent_trades()}
"""

    def _format_recent_trades(self):
        """Format the last 5 trades for readable output"""
        recent_trades = self.trades[-5:] if len(self.trades) >= 5 else self.trades
        formatted_trades = ""
        for trade in recent_trades:
            formatted_trades += f"""
Entry: {trade['entry_time']} at ${trade['entry_price']:.2f}
Exit: {trade['exit_time']} at ${trade['exit_price']:.2f}
P&L: ${trade['pnl']:.2f}
Quantity: {trade['qty']} shares
"""
        return formatted_trades

    async def run_backtest(self):
        """Execute the backtesting simulation using the specified strategy"""
        self.logger.info("Starting backtest...")

        # Get minute-by-minute data
        minute_data = await self.get_minute_data()

        if not minute_data:
            self.logger.error("No data available for backtesting")
            return

        # Initialize trading variables
        position_open = False
        position_qty = 0
        entry_price = None

        # Process each minute
        for row in minute_data:
            timestamp = row['timestamp']
            current_price = float(row['close'])

            if self.in_trading_window(timestamp):
                # Entry logic
                if not position_open:
                    prev_close = float(row['prev_close'])
                    prev_open = float(row['prev_open'])
                    current_open = float(row['Open'])

                    if (prev_close < prev_open) and (current_price < current_open):
                        shares = int(self.TOTAL_USD_AMOUNT / current_price)
                        if shares > 0:
                            position_open = True
                            entry_price = current_price
                            position_qty = shares
                            self.trades.append({
                                'entry_time': timestamp,
                                'entry_price': entry_price,
                                'qty': position_qty,
                                'direction': 'short'
                            })
                            self.logger.info(f"Entered SHORT position: {shares} shares at ${current_price:.2f}")

                # Exit logic - trailing stop
                elif position_open:
                    stop_price = float(entry_price * (1 + self.TRAILING_STOP_PERCENT))
                    if current_price >= stop_price:
                        # Close position
                        pnl = (entry_price - current_price) * position_qty
                        self.cash += pnl
                        self.trades[-1].update({
                            'exit_time': timestamp,
                            'exit_price': current_price,
                            'pnl': pnl
                        })

                        self.logger.info(f"Exited SHORT position: {position_qty} shares at ${current_price:.2f}, P&L: ${pnl:.2f}")

                        position_open = False
                        position_qty = 0
                        entry_price = None

        # Print performance report
        print(self.calculate_performance_metrics())

async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Setup database connection
    db_pool = await asyncpg.create_pool(
        user='postgres',  # Update with your username
        password='your_password',  # Update with your password
        database='tick_data',
        host='localhost'
    )

    try:
        # Get the available data range
        async with db_pool.acquire() as conn:
            data_range = await conn.fetchrow('''
                SELECT 
                    MIN(timestamp) as earliest_tick,
                    MAX(timestamp) as latest_tick
                FROM tick_data
                WHERE ticker = 'COIN'
            ''')

            if data_range['earliest_tick'] is None:
                print("No data found in the tick_data table!")
                return

            # Use the actual data range
            start_date = data_range['earliest_tick']
            end_date = data_range['latest_tick']

        # Create and run backtester
        backtester = MinuteMomentumBacktester(db_pool, start_date, end_date)
        await backtester.run_backtest()

    except Exception as e:
        logging.error(f"Error during backtesting: {e}")
        raise
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
