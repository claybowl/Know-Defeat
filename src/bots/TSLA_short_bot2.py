import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import sys
from pathlib import Path

# # Add the parent directory to sys.path to allow for module imports
# current_dir = Path(__file__).parent
# parent_dir = current_dir.parent
# sys.path.append(str(parent_dir))
# from ib_controller import IBController


class IBClient(EWrapper, EClient):
    """Interactive Brokers client implementation for handling market data and order execution."""
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False

    def connectAck(self):
        self.connected = True
        logging.info("Connected to IB Gateway")

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB Error {errorCode}: {errorString}")


class TSLAShortBot2:
    """Trading bot implementing a momentum-based short strategy for TSLA stock with trailing stop loss."""
    def __init__(self, db_pool, ib_client, bot_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = bot_id
        self.position = None
        self.lowest_price = float('inf')
        self.entry_price = None
        self.current_trade_id = None
        # $10,000 position size for the short position
        self.position_size = 10000
        # Used to trigger a buy-to-cover if price rises a certain % from the lowest point
        self.trailing_stop_pct = 0.002  # 0.2%

    async def get_latest_ticks(self):
        """
        Fetch the last 60 seconds of tick data from the database for TSLA.
        This method returns a DataFrame with columns ['timestamp', 'price'] 
        ordered by timestamp descending.
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    WITH latest_tick AS (
                        SELECT timestamp 
                        FROM tick_data 
                        WHERE ticker = 'TSLA' 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    )
                    SELECT timestamp, price 
                    FROM tick_data 
                    WHERE ticker = 'TSLA'
                    AND timestamp >= (SELECT timestamp - INTERVAL '60 seconds' FROM latest_tick)
                    ORDER BY timestamp DESC;
                """)
            if not rows:
                return None

            # Convert rows to a pandas DataFrame
            df = pd.DataFrame(rows, columns=["timestamp", "price"])
            return df
        except Exception as e:
            self.logger.error(f"Error fetching latest ticks: {e}")
            return None

    def analyze_price_conditions(self, ticks_df):
        """
        Determine if we should initiate a short position based on simple 
        momentum or price trend logic. This example checks if the most recent 
        price is less than the average of the last few ticks.
        """
        if len(ticks_df) < 2:
            return False

        recent_prices = ticks_df['price'][:2]
        current_price = recent_prices.iloc[0]
        prev_price = recent_prices.iloc[1]

        # A simple logic: short if current price < prev price
        # (indicating a short-term downward momentum)
        return current_price < prev_price

    def check_trailing_stop(self, current_price):
        """
        Check if the price has risen enough from the lowest recorded 
        price to trigger the trailing stop for the short position.
        For a short, if the price goes up by more than trailing_stop_pct 
        from the lowest price, exit the position.
        """
        if current_price >= self.lowest_price * (1 + self.trailing_stop_pct):
            self.logger.info(
                f"Trailing stop hit: Price has risen above "
                f"{self.lowest_price * (1 + self.trailing_stop_pct):.2f}"
            )
            return True
        return False

    async def execute_trade(self, side, current_price, timestamp):
        """
        Executes a trade. For short logic:
          - side = "SELL" means open short
          - side = "BUY" means buy-to-cover and exit short
        """
        try:
            # For demonstration only: actual order management would go here.
            self.logger.info(
                f"Executing trade {side} at price {current_price} on {timestamp}"
            )

            if side == "SELL":  # Opening a short position
                self.position = "short"
                self.entry_price = current_price
                self.lowest_price = current_price
                self.current_trade_id = f"short_{timestamp}"
            elif side == "BUY":  # Buying to close short
                self.position = None
                self.entry_price = None
                self.lowest_price = float('inf')
                self.current_trade_id = None

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")

    async def run(self):
        """
        Main bot loop: fetch new ticks, analyze conditions, and either open 
        or close positions based on trailing stops and short signals.
        """
        while True:
            try:
                ticks_df = await self.get_latest_ticks()
                if ticks_df is None or len(ticks_df) == 0:
                    self.logger.info("No tick data available")
                    await asyncio.sleep(1)
                    continue

                current_price = float(ticks_df['price'].iloc[0])
                current_timestamp = ticks_df['timestamp'].iloc[0]
                self.logger.info(f"Processing price: {current_price}")

                if self.position is not None:
                    # Update lowest_price if a new lower price is found
                    if current_price < self.lowest_price:
                        self.lowest_price = current_price

                    if self.check_trailing_stop(current_price):
                        # Buy to close the short
                        await self.execute_trade("BUY", current_price, current_timestamp)
                else:
                    # Evaluate if we should open a new short position
                    if self.analyze_price_conditions(ticks_df):
                        await self.execute_trade("SELL", current_price, current_timestamp)

                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(    
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        db_pool = await asyncpg.create_pool(
            user='clayb',
            password='musicman',
            database='tick_data',
            host='localhost'
        )
        
        ib_client = IBClient()
        bot = TSLAShortBot2(db_pool, ib_client, '3_bot')
        
        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
