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


class COINShortBot2:
    """Trading bot implementing a momentum-based short strategy for COIN with trailing stop loss."""
    def __init__(self, db_pool, ib_client, bot_id):
        """Initialize the trading bot with database connection and parameters."""
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = 4  # fixed bot id for COIN_short_bot2
        self.position = None
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        self.lowest_price = float('inf')
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = 10000  # $10,000 position size 
        self.recent_prices = []
        self.price_buffer_size = 2

    async def get_latest_ticks(self):
        """
        Fetch the last 60 seconds of tick data from the database for COIN.
        Returns a DataFrame with columns ['timestamp', 'price'] 
        ordered by timestamp descending.
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

    def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria for a short trade with a threshold.
        
        For a short trade, we require that the current price is at least 0.1% lower than the price 60s ago
        and that the current price is not higher than the price from 15s ago.
        """
        if ticks_df is None or len(ticks_df) < 2:
            return False
        
        current_price = float(ticks_df['price'].iloc[0])
        price_60s_ago = float(ticks_df['price'].iloc[-1])
        
        # Define a threshold of 0.1% (0.001) for a downward move
        threshold = 0.001
        # Calculate the required maximum price based on the price 60 seconds ago
        required_price = price_60s_ago * (1 - threshold)
        
        # Log key values for debugging
        self.logger.info(f"Current price: {current_price}, Price 60s ago: {price_60s_ago}, Required maximum price: {required_price:.4f}")
        
        # Determine the price from approximately 15 seconds ago
        latest_time = ticks_df['timestamp'].iloc[0]
        cutoff_time = latest_time - timedelta(seconds=15)
        ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
        if len(ticks_15s_ago) == 0:
            return False
        price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
        self.logger.info(f"Price 15s ago: {price_15s_ago}")
        
        # For a short trade, the entry condition is fulfilled if the current price is less than or equal to
        # both the required maximum price (i.e., reflecting at least a 0.1% drop from 60s ago) and the
        # price from 15 seconds ago.
        if current_price <= required_price and current_price <= price_15s_ago:
            return True
        return False

    def check_trailing_stop(self, current_price):
        """
        Check if trailing stop has been hit for short position.
        For a short position, if price rises by trailing_stop_pct from the lowest
        recorded price, exit the position (buy to cover).
        """
        if self.position is None:
            return False

        if current_price < self.lowest_price:
            self.lowest_price = current_price

        stop_price = self.lowest_price * (1 + self.trailing_stop_pct)
        
        return current_price >= stop_price

    async def execute_trade(self, action, price, timestamp):
        """
        Execute a trade (entry or exit) for a short position on COIN.
        'SELL' action for opening the short, and 'BUY' to cover (close the short).
        """
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            if action.upper() == "SELL":
                # Opening a short position
                self.logger.info(f"Initiating SHORT SELL at ${price:.2f}")
                self.position = 'SHORT'
                self.entry_price = price
                self.lowest_price = price
                await self.log_trade_entry(price, timestamp)

            elif action.upper() == "BUY":
                # Covering the short position
                self.logger.info(f"Covering SHORT position at ${price:.2f}")
                await self.log_exit_signal(price, timestamp)
                await self.log_trade_exit(price, timestamp)
                self.position = None
                self.entry_price = None
                self.lowest_price = float('inf')
            else:
                self.logger.error(f"Unknown trade action: {action}")

        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")
            raise

    async def log_trade_entry(self, price, timestamp):
        """Log trade entry to the database."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO sim_bot_trades 
                    (entry_time, ticker, entry_price, trade_direction, 
                     trade_size, trade_status, bot_id)
                    VALUES ($1, 'COIN', $2, 'SHORT', $3, 'open', $4)
                    RETURNING trade_id
                """, timestamp, price, self.position_size, self.bot_id)
                
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
        """Log actual trade exit details."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sim_bot_trades 
                    SET exit_price = $1,
                        exit_time = $2,
                        trade_pnl = entry_price - $1,
                        trade_status = 'closed'
                    WHERE trade_id = $3
                """, price, timestamp, self.current_trade_id)
        except Exception as e:
            self.logger.error(f"Error in log_trade_exit: {e}")
            raise

    async def run(self):
        """Main bot loop for COIN short strategy."""
        self.logger.info("Starting COIN Short Bot...")

        # Connect to IB
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

                # If we have an open short position, check trailing stop
                if self.position is not None:
                    if self.check_trailing_stop(current_price):
                        await self.execute_trade("BUY", current_price, ticks_df['timestamp'].iloc[0])
                else:
                    # If no position, see if short conditions are met
                    if self.analyze_price_conditions(ticks_df):
                        await self.execute_trade("SELL", current_price, ticks_df['timestamp'].iloc[0])

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
        # Pass integer bot_id instead of string
        bot = COINShortBot2(db_pool, ib_client, 7)

        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
