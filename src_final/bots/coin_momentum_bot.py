import sys
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import asyncpg
from ibapi.contract import Contract
from ibapi.order import Order
from utils.time_utils import is_market_hours, get_current_minute_start

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coin_momentum_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class CoinMomentumBot:
    def __init__(self, db_pool, ib_client):
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.position = None
        self.entry_price = None
        self.trailing_stop_price = None
        self.position_size = 10000  # $10,000 position size
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        # Adding a price history buffer to store recent ticks
        self.recent_prices = []
        self.price_buffer_size = 2  # We only need to compare two prices

    async def process_tick(self, ticker, price, timestamp):
        """Process each new tick of data"""
        if not is_market_hours() or ticker != 'COIN':
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
                if (self.position is None and 
                    first_tick['price'] < last_tick['price']):

                    # Calculate quantity based on position size
                    quantity = int(self.position_size / price)

                    # Enter long position
                    contract = self.create_coin_contract()
                    order = self.create_order("BUY", quantity)
                    self.ib_client.placeOrder(self.ib_client.next_valid_order_id, contract, order)

                    self.position = "LONG"
                    self.entry_price = price
                    self.trailing_stop_price = price * (1 - self.trailing_stop_pct)  # Stop loss below entry for longs
                    self.logger.info(f"Entered LONG position: {quantity} shares at ${price:.2f}")
                    self.logger.info(f"Price movement that triggered entry: {first_tick['price']:.2f} -> {last_tick['price']:.2f}")

                # Check exit conditions if in position
                elif self.position is not None:
                    if await self.check_and_update_trailing_stop(price):
                        # Exit position
                        quantity = int(self.position_size / self.entry_price)
                        contract = self.create_coin_contract()
                        order = self.create_order("SELL", quantity)
                        self.ib_client.placeOrder(self.ib_client.next_valid_order_id, contract, order)

                        pnl = (price - self.entry_price) * quantity  # Modified for long positions
                        self.logger.info(f"Exited LONG position: {quantity} shares at ${price:.2f}, P&L: ${pnl:.2f}")

                        # Reset position tracking
                        self.position = None
                        self.entry_price = None
                        self.trailing_stop_price = None
                        self.recent_prices = []  # Clear price buffer after exit

        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")

    async def check_and_update_trailing_stop(self, current_price):
        """Update trailing stop price and check if it's triggered"""
        if self.position is None:
            return False

        if self.position == "LONG":
            # For long positions, trailing stop is below entry
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                self.logger.info(f"Updated trailing stop to: ${self.trailing_stop_price:.2f}")

            # Check if stop is triggered
            if current_price <= self.trailing_stop_price:
                return True

        return False
