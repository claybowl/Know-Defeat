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
import time


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
        self.bot_id = 8  # fixed bot id for TSLA_short_bot2
        self.position = None
        self.lowest_price = float('inf')
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = 10000  # $10,000 position size
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        # Add tracking for current price and minute OHLC
        self.current_price = None
        self.last_minute_open = None
        self.last_minute_close = None
        self.last_minute_high = float('-inf')
        self.last_minute_low = float('inf')

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

    # We need to add logging throughout the following functions so we can see the logic of the minute creation and the short entry logic.
    # We need win rate and average PnL of bot_id. 

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

    async def process_tick(self, tick):
        """
        Handle a single tick of data. Updates current price and minute OHLC data.
        Implements a simple momentum strategy based on minute close < open for shorts.
        """
        try:
            # Update current price
            self.current_price = float(tick['price'])
            self.logger.debug(f"Current price updated to: {self.current_price:.2f}")

            # Update minute OHLC data
            if tick['timestamp'].second == 0:  # Start of new minute
                self.logger.debug("New minute started")
                self.last_minute_open = self.current_price
                self.last_minute_high = self.current_price
                self.last_minute_low = self.current_price
            else:
                if self.current_price > self.last_minute_high:
                    self.last_minute_high = self.current_price
                    self.logger.debug(f"New minute high updated to: {self.last_minute_high:.2f}")
                if self.current_price < self.last_minute_low:
                    self.last_minute_low = self.current_price
                    self.logger.debug(f"New minute low updated to: {self.last_minute_low:.2f}")
                if tick['timestamp'].second == 59:  # End of minute
                    self.last_minute_close = self.current_price
                    self.logger.debug(f"Minute closed with price: {self.last_minute_close:.2f}")

            # Check trading conditions for short entry
            if self.last_minute_close and self.last_minute_open:
                if self.last_minute_close < self.last_minute_open and self.current_price < self.last_minute_open:
                    self.logger.info("Short entry condition met")
                    quantity = self.position_size / self.current_price
                    await self.place_order('TSLA', quantity, 'SELL')

            # Check trailing stop if in position
            if self.position is not None and self.check_trailing_stop(self.current_price):
                self.logger.info("Trailing stop condition met")
                quantity = self.position_size / self.current_price
                await self.place_order('TSLA', quantity, 'BUY')

            self.logger.debug(f"Processed tick: {tick['timestamp']} Price: {self.current_price:.2f} Volume: {tick['volume']}")

        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
            raise

    async def place_order(self, ticker, quantity, side):
        """Place an order with the specified parameters."""
        try:
            if self.current_price is None:
                self.logger.error("Cannot place order: current price is not set")
                return

            self.logger.info(f"Placing {side} order for {ticker}: {quantity:.4f} shares at {self.current_price:.2f}")
            
            # Execute the trade using our existing logic
            await self.execute_trade(side, self.current_price, datetime.now())
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise

    async def run(self):
        """Main bot loop."""
        self.logger.info("Starting TSLA Short Bot...")
        
        # Add connection retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.ib_client.connect('127.0.0.1', 4002, 1)
                
                # Wait for connection with timeout
                connection_timeout = 10  # seconds
                start_time = time.time()
                
                while not self.ib_client.connected:
                    if time.time() - start_time > connection_timeout:
                        raise TimeoutError("Connection timeout to IB Gateway")
                    await asyncio.sleep(0.1)
                
                self.logger.info("Connected to Interactive Brokers")
                break  # Successfully connected
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(5)  # Wait before retrying
                else:
                    raise RuntimeError("Failed to connect to IB Gateway after maximum retries")

        # Main trading loop
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

    async def log_trade_exit(self, price, timestamp):
        """Log actual trade exit details."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                # First get the entry price and trade size
                trade = await conn.fetchrow("""
                    SELECT entry_price, trade_size 
                    FROM sim_bot_trades 
                    WHERE trade_id = $1
                """, self.current_trade_id)
                
                if trade:
                    entry_price = trade['entry_price']
                    trade_size = trade['trade_size']
                    
                    # Calculate number of shares based on trade size and entry price
                    shares = trade_size / entry_price
                    # For short positions, profit is when exit price is lower than entry
                    trade_pnl = shares * (entry_price - price)
                    
                    await conn.execute("""
                        UPDATE sim_bot_trades 
                        SET exit_price = $1,
                            exit_time = $2,
                            trade_pnl = $3,
                            trade_status = 'closed'
                        WHERE trade_id = $4
                    """, price, timestamp, trade_pnl, self.current_trade_id)
        except Exception as e:
            self.logger.error(f"Error in log_trade_exit: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        try:
            db_pool = await asyncpg.create_pool(
                user='clayb',
                password='musicman',
                database='tick_data',
                host='localhost'
            )
            
            ib_client = IBClient()
            bot = TSLAShortBot2(db_pool, ib_client, 8)
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
