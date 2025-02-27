'''\nCOIN_long_bot.py\n\nThis module implements the CoinLongBot, a trading bot that applies a momentum-based long strategy on the 'COIN' ticker. It uses the IB API to connect to Interactive Brokers and asyncpg to interact with a PostgreSQL database.\n\nClasses:\n- IBClient: Manages the connection to Interactive Brokers and handles error logging.\n- CoinLongBot: Contains methods to fetch tick data from the database, analyze price conditions for trade entries, check for trailing stops, execute trades, and log trade details.\nEach method includes inline comments explaining the key steps in the processing logic.\n'''

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pandas as pd  # type: ignore
from decimal import Decimal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class IBClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False

    def connectAck(self):
        self.connected = True
        logging.info("Connected to IB Gateway")

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB Error {errorCode}: {errorString}")

class CoinLongBot:

    def __init__(self, db_pool, ib_client, bot_id):
        """Initialize the trading bot with database connection and parameters."""
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = 1  # fixed bot id for COIN_long_bot
        self.algo_id = 1  # This bot belongs to algo_id 1
        self.position = None
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        self.highest_price = 0
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = 10000  # $10,000 position size 
        self.trailing_stop = 0.002  # 0.2%
        self.recent_prices = []
        self.price_buffer_size = 2  # We only need to compare two prices

    def setup_logger(self):
        """Configure logging for the bot."""
        logger = logging.getLogger('CoinMomentumBot')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def get_latest_ticks(self):
        """Fetch the last 60 seconds of tick data from the database."""
        # Query the tick_data table for the most recent records for 'COIN'.
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

                # Convert the result into a Pandas DataFrame
                df = pd.DataFrame(rows, columns=['timestamp', 'price'])

                if not df.empty:
                    self.logger.info(f"Latest price: {df['price'].iloc[0]}")
                    self.logger.info(f"Oldest price: {df['price'].iloc[-1]}")

                return df

        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None

    async def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria with a threshold.
        
        To trigger a long trade, we require that the current price has increased
        by at least a small percentage (e.g., 0.1%) compared to the price 60 seconds ago,
        and that the current price is at least as high as the price from 15 seconds ago.
        """
        if ticks_df is None or len(ticks_df) < 2:
            return False

        # Get the current price and the price 60 seconds ago from the tick data
        current_price = float(ticks_df['price'].iloc[0])
        price_60s_ago = float(ticks_df['price'].iloc[-1])

        # Define a threshold of 0.1% (0.001) for the required price increase
        threshold = 0.001
        # Calculate the percentage gain from 60 seconds ago
        percent_gain = (current_price - price_60s_ago) / price_60s_ago
        self.logger.info(f"Current price: {current_price}, Price 60s ago: {price_60s_ago}, Percent gain: {percent_gain:.4f}")

        # Determine the price from roughly 15 seconds ago for additional momentum confirmation
        latest_time = ticks_df['timestamp'].iloc[0]
        cutoff_time = latest_time - timedelta(seconds=15)
        ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
        if len(ticks_15s_ago) == 0:
            return False
        price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
        self.logger.info(f"Price 15s ago: {price_15s_ago}")

        # Modified entry condition: check if the percentage gain is at least the threshold
        # and ensure the current price is not lower than the price from 15 seconds ago
        if percent_gain >= threshold and current_price >= price_15s_ago:
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
        """Execute a trade order with enhanced exit tracking."""
        # Execute BUY to enter position and SELL to exit, while logging trade details and updating bot state
        try:
            if action == "BUY":
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                self.logger.info(f"BUY executed at {price}")
                await self.log_trade_entry(price, timestamp)

            elif action == "SELL":
                pnl = (price - self.entry_price) / self.entry_price * 100
                self.logger.info(f"SELL signal at {price}. PnL: {pnl:.2f}%")

                # Log the exit signal then the actual trade exit
                await self.log_exit_signal(price, timestamp)
                actual_exit_price = price 
                actual_exit_time = timestamp 
                await self.log_trade_exit(actual_exit_price, actual_exit_time)

                # Reset position state after trade exit
                self.position = None
                self.highest_price = 0
                self.entry_price = None

        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")
            raise

    async def log_trade_entry(self, price, timestamp):
        """Log trade entry to the database."""
        # Insert a new record for the trade entry into the sim_bot_trades table
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO sim_bot_trades 
                    (entry_time, ticker, entry_price, trade_direction,
                     trade_size, trade_status, bot_id, algo_id)
                    VALUES ($1, 'COIN', $2, 'LONG', $3, 'open', $4, $5)
                    RETURNING trade_id
                """, timestamp, price, self.position_size, self.bot_id, self.algo_id)
                
                if result:
                    self.current_trade_id = result['trade_id']

        except Exception as e:
            self.logger.error(f"Error in log_trade_entry: {e}")
            raise

    async def log_exit_signal(self, price, timestamp):
        """Log when exit conditions are first met."""
        # Update the existing trade record to mark the exit trigger conditions
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
        # Finalize the trade record by logging the exit price, time, and calculating profit/loss
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
        # Connect to IB, continuously fetch tick data, analyze for trade signals, and execute trades accordingly.
        self.logger.info("Starting COIN Momentum Bot...")

        # Establish connection with Interactive Brokers
        self.ib_client.connect('127.0.0.1', 4002, 1)  # Use 7496 for paper trading

        # Wait until connection is confirmed
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

                # If already in a position, check if trailing stop conditions are met to exit
                if self.position is not None:
                    if self.check_trailing_stop(current_price):
                        await self.execute_trade("SELL", current_price, ticks_df['timestamp'].iloc[0])
                # If no active position, check for entry conditions
                else:
                    if self.analyze_price_conditions(ticks_df):
                        await self.execute_trade("BUY", current_price, ticks_df['timestamp'].iloc[0])

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    async def main():
        # Create the database pool
        db_pool = await asyncpg.create_pool(
            user='clayb',
            password='musicman',
            database='tick_data',
            host='localhost'
        )

        # Initialize the IB client
        ib_client = IBClient()

        # Create an instance of CoinLongBot
        bot = CoinLongBot(db_pool, ib_client, 1)

        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
