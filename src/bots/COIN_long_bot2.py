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


class COINLongBot2:
    """Trading bot implementing a momentum-based long strategy for COIN with trailing stop loss."""
    def __init__(self, db_pool, ib_client, bot_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = 3  # fixed bot id for COIN_long_bot2
        self.algo_id = 2  # This bot belongs to algo_id 2
        self.position = None
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        self.highest_price = 0
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = 10000  # $10,000 position size 
        self.trailing_stop_price = None
        self.recent_prices = []
        self.price_buffer_size = 2

    async def get_latest_ticks(self):
        """
        Fetch the last 60 seconds of tick data from the database for COIN.
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    WITH latest_tick AS (
                        SELECT timestamp 
                        FROM tick_data 
                        WHERE ticker = 'COIN' 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    )
                    SELECT timestamp, price 
                    FROM tick_data 
                    WHERE ticker = 'COIN'
                    AND timestamp >= (SELECT timestamp - INTERVAL '60 seconds' FROM latest_tick)
                    ORDER BY timestamp DESC;
                """)
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=["timestamp", "price"])
            return df
        except Exception as e:
            self.logger.error(f"Error fetching latest ticks: {e}")
            return None

    async def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria for a long trade with a threshold.
        
        For a long trade, we require that the current price is at least 0.1% higher than the price 60 seconds ago
        and that the current price is not lower than the price from 15 seconds ago. This extra threshold helps
        to ensure that only meaningful upward movements trigger a trade.
        """
        if ticks_df is None or len(ticks_df) < 2:
            return False
        
        # Extract current price and the price from 60 seconds ago
        current_price = float(ticks_df['price'].iloc[0])
        price_60s_ago = float(ticks_df['price'].iloc[-1])
        
        # Define the threshold (0.1% increase)
        threshold = 0.001
        # Compute the minimum required price based on the price 60 seconds ago
        required_price = price_60s_ago * (1 + threshold)
        
        # Log values for debugging purposes
        self.logger.info(f"[COIN_long_bot2] Current price: {current_price}, Price 60s ago: {price_60s_ago}, Required minimum price: {required_price:.4f}")
        
        # Determine the price from approximately 15 seconds ago
        latest_time = ticks_df['timestamp'].iloc[0]
        cutoff_time = latest_time - timedelta(seconds=15)
        ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
        if len(ticks_15s_ago) == 0:
            return False
        price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
        self.logger.info(f"[COIN_long_bot2] Price 15s ago: {price_15s_ago}")
        
        # Check if the current price meets both conditions: 
        # 1. current price is at least the required minimum (reflecting a 0.1% upward gain over the price 60s ago)
        # 2. current price is not lower than the price from 15 seconds ago
        if current_price >= required_price and current_price >= price_15s_ago:
            return True
        return False

    def check_trailing_stop(self, current_price):
        """
        Check if trailing stop has been hit. For a long position, if price falls below
        a certain percentage (trailing_stop_pct) from the highest recorded price, exit the position.
        """
        if self.position is None:
            return False

        if current_price > self.highest_price:
            self.highest_price = current_price

        stop_price = self.highest_price * (1 - self.trailing_stop_pct)

        return current_price <= stop_price

    async def execute_trade(self, action, price, timestamp):
        """
        Execute a trade order with logging for entry/exit events.
          BUY - open a long position
          SELL - close the long position
        """
        try:
            if action == "BUY":
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                self.logger.info(f"BUY executed at {price}")
                await self.log_trade_entry(price, timestamp)

            elif action == "SELL":
                # Removed PnL calculation. Use a simpler log message instead.
                self.logger.info(f"SELL signal at {price}.")

                await self.log_exit_signal(price, timestamp)
                actual_exit_price = price
                actual_exit_time = timestamp
                await self.log_trade_exit(actual_exit_price, actual_exit_time)

                self.position = None
                self.highest_price = 0
                self.entry_price = None

        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")
            raise

    async def log_trade_entry(self, price, timestamp):
        """Log trade entry to the database."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            # Convert bot_id string to integer
            numeric_bot_id = self.bot_id

            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO sim_bot_trades 
                    (entry_time, ticker, entry_price, trade_direction, 
                     trade_size, trade_status, bot_id, algo_id)
                    VALUES ($1, 'COIN', $2, 'LONG', $3, 'open', $4, $5)
                    RETURNING trade_id
                """, timestamp, price, self.position_size, numeric_bot_id, self.algo_id)

                if result:
                    self.current_trade_id = result['trade_id']

        except Exception as e:
            self.logger.error(f"Error in log_trade_entry: {e}")
            raise

    async def log_exit_signal(self, price, timestamp):
        """Log when exit conditions are first met."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sim_bot_trades 
                    SET exit_trigger_price = $1,
                        exit_trigger_time = $2,
                        trade_status = 'pending_exit'
                    WHERE trade_id = $3
                """, price, timestamp, self.current_trade_id)

        except Exception as e:
            self.logger.error(f"Error in log_exit_signal: {e}")
            raise

    async def log_trade_exit(self, price, timestamp):
        """
        Log the final execution details of the exit.
        """
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sim_bot_trades 
                    SET exit_price = $1,
                        exit_time = $2,
                        trade_pnl = $1 - entry_price,
                        trade_status = 'closed'
                    WHERE trade_id = $3
                """, price, timestamp, self.current_trade_id)
        except Exception as e:
            self.logger.error(f"Error in log_trade_exit: {e}")
            raise

    async def run(self):
        """Main bot loop."""
        self.logger.info("Starting COIN Long Bot...")

        self.ib_client.connect('127.0.0.1', 4002, 1)
        
        while not self.ib_client.connected:
            await asyncio.sleep(0.1)

        self.logger.info("Connected to Interactive Brokers")

        while True:
            try:
                ticks_df = await self.get_latest_ticks()
                if ticks_df is None or len(ticks_df) == 0:
                    self.logger.info("No tick data available")
                    await asyncio.sleep(1)
                    continue

                current_price = float(ticks_df['price'].iloc[0])
                self.logger.info(f"Processing price: {current_price}")

                if self.position is not None:
                    if self.check_trailing_stop(current_price):
                        await self.execute_trade("SELL", current_price, ticks_df['timestamp'].iloc[0])
                else:
                    if self.analyze_price_conditions(ticks_df):
                        await self.execute_trade("BUY", current_price, ticks_df['timestamp'].iloc[0])

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
        bot = COINLongBot2(db_pool, ib_client, '3')

        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
