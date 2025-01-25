import sys
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import asyncpg
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.client import EClient
from ibapi.wrapper import EWrapper


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coin_momentum_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class YourIBClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        # Additional initialization code

class CoinMomentumBot:
    """Coin Momentum Bot class"""
    def __init__(self, db_pool, ib_client, bot_id):
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = bot_id
        self.position = None
        self.entry_price = None
        self.trailing_stop_price = None
        self.position_size = 10000  # $10,000 position size 
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        # Adding a price history buffer to store recent ticks
        self.recent_prices = []
        self.price_buffer_size = 2  # We only need to compare two prices

    def create_coin_contract(self):
        """Create and return a contract object for COIN stock."""
        contract = Contract()
        contract.symbol = "COIN"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.primaryExchange = "NASDAQ"  # Set primary exchange for NASDAQ stocks
        return contract

    def create_order(self, action, quantity):
        """Create and return an order object."""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "MKT"  # Market order
        return order

    async def process_tick(self, ticker, price, timestamp):
        """Process each new tick of data"""
        if ticker != 'COIN':
            return

        try:
            # Add new price to our buffer with its timestamp
            self.recent_prices.append({'price': price, 'timestamp': timestamp})

            # Keep only the most recent prices within our buffer size
            if len(self.recent_prices) > self.price_buffer_size:
                self.recent_prices.pop(0)  # Remove oldest price

            # We need at least 2 prices to make a comparison
            if len(self.recent_prices) == self.price_buffer_size:
                # Get first and last price from our buffer
                first_tick = self.recent_prices[0]
                last_tick = self.recent_prices[-1]

                # Check entry conditions - price is rising
                if self.position is None and first_tick['price'] < last_tick['price']:
                    quantity = int(self.position_size / price)  

                    # Enter long position
                    contract = self.create_coin_contract()
                    order = self.create_order("BUY", quantity)
                    self.ib_client.placeOrder(self.ib_client.next_valid_order_id, contract, order)

                    # Store trade details
                    self.position = "LONG"
                    self.entry_price = price
                    self.entry_timestamp = timestamp  
                    self.quantity = quantity
                    self.trailing_stop_price = price * (1 - self.trailing_stop_pct)

                    self.logger.info("Entered LONG position: %d shares at $%.2f", quantity, price)
                    self.logger.info("Price movement that triggered entry: %.2f -> %.2f", first_tick['price'], last_tick['price'])

                    # Insert trade into sim_bot_trades
                    await self.insert_trade(timestamp, ticker, price, quantity)

                # Check exit conditions if in position
                elif self.position is not None:
                    if await self.check_and_update_trailing_stop(price):
                        # Exit position
                        contract = self.create_coin_contract()
                        order = self.create_order("SELL", self.quantity)
                        self.ib_client.placeOrder(self.ib_client.next_valid_order_id, contract, order)

                        pnl = (price - self.entry_price) * self.quantity
                        trade_duration = timestamp - self.entry_timestamp
                        self.logger.info("Exited LONG position: %d shares at $%.2f, P&L: $%.2f", self.quantity, price, pnl)

                        # Update trade in sim_bot_trades with exit details
                        await self.update_trade(price, pnl, trade_duration)

                        # Update bot_metrics
                        await self.update_bot_metrics(pnl)

                        # Reset position tracking
                        self.position = None
                        self.entry_price = None
                        self.trailing_stop_price = None
                        self.recent_prices = []

        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")

    async def check_and_update_trailing_stop(self, current_price):
        """Update trailing stop price and check if it's triggered"""  
        if self.position == "LONG":
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > self.trailing_stop_price:  
                self.trailing_stop_price = new_stop
                self.logger.info("Updated trailing stop to: $%.2f", self.trailing_stop_price)

            if current_price <= self.trailing_stop_price:
                return True
        return False

    async def insert_trade(self, timestamp, ticker, entry_price, quantity):
        """Insert new trade into sim_bot_trades"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sim_bot_trades (bot_id, trade_timestamp, symbol, 
                            entry_price, quantity, trade_type, trade_direction)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, self.bot_id, timestamp, ticker, entry_price, quantity, "BUY", "LONG")

    async def update_trade(self, exit_price, pnl, trade_duration):  
        """Update trade in sim_bot_trades with exit details"""
        async with self.db_pool.acquire() as conn:  
            await conn.execute("""
                UPDATE sim_bot_trades
                SET exit_price = $1, profit_loss = $2, trade_duration = $3  
                WHERE bot_id = $4
                ORDER BY id DESC 
                LIMIT 1
            """, exit_price, pnl, trade_duration, self.bot_id)

    async def update_bot_metrics(self, trade_pnl):
        """Update bot_metrics with latest stats after each trade"""
        async with self.db_pool.acquire() as conn:
            # Get stats 
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) AS winning_trades,
                    COALESCE(SUM(profit_loss), 0) AS total_pnl  
                FROM sim_bot_trades
                WHERE bot_id = $1
            """, self.bot_id)

            # Calculate metrics
            total_trades = row['total_trades'] 
            total_pnl = row['total_pnl']
            avg_win_rate = row['winning_trades'] / total_trades if total_trades else 0

            win_streak = await conn.fetchval("""
                SELECT MAX(count) FROM (
                    SELECT COUNT(*) as count  
                    FROM (
                        SELECT profit_loss, 
                               SUM(CASE WHEN profit_loss > 0 THEN 0 ELSE 1 END) OVER (
                                 ORDER BY id DESC
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  
                               ) AS period
                        FROM sim_bot_trades  
                        WHERE bot_id = $1
                    ) t1
                    GROUP BY period  
                ) t2
            """, self.bot_id)

            # Update bot_metrics
            await conn.execute("""
                INSERT INTO bot_metrics (bot_id, avg_win_rate, total_pnl, 
                            win_streak_2, win_streak_3, win_streak_4,
                            win_streak_5, win_streak_6, win_streak_7)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (bot_id) DO UPDATE 
                SET avg_win_rate = EXCLUDED.avg_win_rate,
                    total_pnl = EXCLUDED.total_pnl,  
                    win_streak_2 = EXCLUDED.win_streak_2,
                    win_streak_3 = EXCLUDED.win_streak_3, 
                    win_streak_4 = EXCLUDED.win_streak_4,
                    win_streak_5 = EXCLUDED.win_streak_5,
                    win_streak_6 = EXCLUDED.win_streak_6, 
                    win_streak_7 = EXCLUDED.win_streak_7
            """, self.bot_id, avg_win_rate, total_pnl,  
                win_streak>=2, win_streak>=3, win_streak>=4,
                win_streak>=5, win_streak>=6, win_streak>=7)

        # Log the metrics
        self.logger.info(f"Total Trades: {total_trades}")
        self.logger.info(f"Total PnL: {total_pnl}")
        self.logger.info("Average Win Rate: %.2f%%", avg_win_rate * 100)
        self.logger.info(f"Win Streak: {win_streak}")

        # Print the metrics to the terminal
        print(f"Total Trades: {total_trades}")
        print(f"Total PnL: {total_pnl}")
        print("Average Win Rate: %.2f%%", avg_win_rate * 100)
        print(f"Win Streak: {win_streak}")

async def main():
    # Initialize necessary components like db_pool and ib_client
    db_pool = await asyncpg.create_pool(dsn='postgresql://clayb:musicman@localhost:5432/tick_data')
    ib_client = YourIBClient()  # Replace with the actual initialization

    # Create an instance of CoinMomentumBot
    bot = CoinMomentumBot(db_pool, ib_client, bot_id='your_bot_id')

    # Example of processing a tick
    await bot.process_tick('COIN', 250.0, datetime.now())

if __name__ == "__main__":
    asyncio.run(main())
