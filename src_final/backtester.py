import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pytz
from decimal import Decimal
import sys
from pathlib import Path


# Add the parent directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from bots.coin_momentum_bot import CoinMomentumBot


class MockIBClient:
    """Mock IB client for backtesting that simulates order placement"""
    def __init__(self):
        self.next_valid_order_id = 1
        self.orders = []
        self.logger = logging.getLogger(__name__)

    def placeOrder(self, orderId, contract, order):
        """Simulate order placement by logging the order details"""
        self.orders.append({
            'orderId': orderId,
            'symbol': contract.symbol,
            'action': order.action,
            'quantity': order.totalQuantity,
            'timestamp': datetime.now(pytz.UTC)
        })
        self.logger.info(f"Backtester Order Placed: {order.action} {order.totalQuantity} {contract.symbol}")
        self.next_valid_order_id += 1

class CoinBacktester:
    """Backtesting system for the COIN momentum strategy"""

    def __init__(self, db_pool, start_date, end_date):
        self.db_pool = db_pool
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)

        # Create mock IB client for backtesting
        self.mock_ib = MockIBClient()

        # Initialize performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_equity = 0
        self.trades = []

    async def get_historical_ticks(self):
        """Fetch historical tick data from the database with detailed verification"""
        try:
            async with self.db_pool.acquire() as conn:
                # First, let's check how many ticks exist in our date range
                count = await conn.fetchval('''
                    SELECT COUNT(*)
                    FROM tick_data
                    WHERE 
                        ticker = 'COIN'
                        AND timestamp BETWEEN $1 AND $2
                ''', self.start_date, self.end_date)

                self.logger.info(f"Found {count} total ticks in the specified date range")

                # Get time range information
                time_range = await conn.fetchrow('''
                    SELECT 
                        MIN(timestamp) as earliest_tick,
                        MAX(timestamp) as latest_tick,
                        COUNT(DISTINCT DATE(timestamp)) as unique_days
                    FROM tick_data
                    WHERE 
                        ticker = 'COIN'
                        AND timestamp BETWEEN $1 AND $2
                ''', self.start_date, self.end_date)

                self.logger.info(f"""
    Data Range Summary:
    ------------------
    Earliest tick: {time_range['earliest_tick']}
    Latest tick: {time_range['latest_tick']}
    Number of trading days: {time_range['unique_days']}
                """)

                # Get sample of data to verify content
                sample_data = await conn.fetch('''
                    SELECT 
                        ticker,
                        timestamp,
                        price,
                        volume
                    FROM tick_data
                    WHERE 
                        ticker = 'COIN'
                        AND timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp ASC
                    LIMIT 5
                ''', self.start_date, self.end_date)

                self.logger.info("\nFirst 5 ticks in the dataset:")
                for tick in sample_data:
                    self.logger.info(f"Time: {tick['timestamp']}, Price: ${tick['price']}, Volume: {tick['volume']}")

                # Now get all the actual data for backtesting
                ticks = await conn.fetch('''
                    SELECT 
                        ticker,
                        timestamp,
                        price,
                        volume
                    FROM tick_data
                    WHERE 
                        ticker = 'COIN'
                        AND timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp ASC
                ''', self.start_date, self.end_date)

                self.logger.info(f"\nSuccessfully loaded {len(ticks)} ticks for backtesting")

                # Verify data continuity
                if len(ticks) > 1:
                    avg_time_between_ticks = (ticks[-1]['timestamp'] - ticks[0]['timestamp']) / len(ticks)
                    self.logger.info(f"Average time between ticks: {avg_time_between_ticks}")

                return ticks

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            self.logger.error(f"Error details: ", exc_info=True)
            return []

    def calculate_metrics(self, trade):
        """Calculate and update performance metrics for each trade"""
        self.total_trades += 1
        self.total_pnl += trade['pnl']

        if trade['pnl'] > 0:
            self.winning_trades += 1

        # Update peak equity and drawdown
        if self.total_pnl > self.peak_equity:
            self.peak_equity = self.total_pnl
        else:
            current_drawdown = self.peak_equity - self.total_pnl
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

    def print_performance_report(self):
        """Generate and print detailed performance metrics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0

        print("\n=== Backtesting Results ===")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: ${self.total_pnl:,.2f}")
        print(f"Average P&L per Trade: ${avg_pnl:,.2f}")
        print(f"Maximum Drawdown: ${self.max_drawdown:,.2f}")
        print("\nTrade Log:")
        for trade in self.trades:
            print(f"Entry: {trade['entry_time']} | Exit: {trade['exit_time']} | "
                  f"P&L: ${trade['pnl']:,.2f} | Duration: {trade['duration']}")

    async def run_backtest(self):
        """Execute the backtest using historical data"""
        # Initialize the bot with mock IB client

        bot = CoinMomentumBot(self.db_pool, self.mock_ib)

        self.logger.info("Starting backtest...")

        # Get historical data
        ticks = await self.get_historical_ticks()
        total_ticks = len(ticks)
        self.logger.info(f"Loaded {total_ticks} ticks for backtesting")

        # Process each tick
        for tick in ticks:
            # Process the tick through the bot
            await bot.process_tick(tick['ticker'], tick['price'], tick['timestamp'])

            # If we have a completed trade, record it
            if len(self.mock_ib.orders) >= 2:  # Entry and exit make a complete trade
                entry_order = self.mock_ib.orders.pop(0)
                exit_order = self.mock_ib.orders.pop(0)

                trade = {
                    'entry_time': entry_order['timestamp'],
                    'exit_time': exit_order['timestamp'],
                    'duration': exit_order['timestamp'] - entry_order['timestamp'],
                    'entry_price': tick['price'],  # Simplified - in reality would use order price
                    'exit_price': tick['price'],
                    'quantity': entry_order['quantity'],
                    'pnl': (entry_order['quantity'] * 
                           (tick['price'] - tick['price']))  # Simplified P&L calculation
                }

                self.trades.append(trade)
                self.calculate_metrics(trade)

        # Generate performance report
        self.print_performance_report()

async def main():
    """Main function to run the backtest"""
        # Setup logging to see detailed output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Setup database connection
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )

    # First, let's find out what data we actually have
    async with db_pool.acquire() as conn:
        data_range = await conn.fetchrow('''
            SELECT 
                MIN(timestamp) as earliest_tick,
                MAX(timestamp) as latest_tick,
                COUNT(*) as total_ticks
            FROM tick_data
            WHERE ticker = 'COIN'
        ''')

        if data_range['total_ticks'] == 0:
            print("No data found in the tick_data table!")
            return

        print(f"\nAvailable Data Range:")
        print(f"Earliest tick: {data_range['earliest_tick']}")
        print(f"Latest tick: {data_range['latest_tick']}")
        print(f"Total ticks: {data_range['total_ticks']}")

        # Use the actual data range we found
        start_date = data_range['earliest_tick']
        end_date = data_range['latest_tick']

    # Create and run backtester
    backtester = CoinBacktester(db_pool, start_date, end_date)
    await backtester.run_backtest()

    # Clean up
    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
