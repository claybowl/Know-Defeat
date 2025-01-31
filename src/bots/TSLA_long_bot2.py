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


class TSLALongBot2:
    """Trading bot implementing a momentum-based long strategy for TSLA stock with trailing stop loss."""
    def __init__(self, db_pool, ib_client, bot_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = bot_id
        self.position = None
        self.trailing_stop = 0.002  # 0.2%
        self.highest_price = 0
        self.entry_price = None
        self.current_trade_id = None
        self.trailing_stop_price = None
        self.position_size = 10000  # $10,000 position size 
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        self.recent_prices = []
        self.price_buffer_size = 2

    async def get_latest_ticks(self):
        """Fetch the last 60 seconds of tick data from the database."""
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
                
                df = pd.DataFrame(rows, columns=['timestamp', 'price'])
                
                if not df.empty:
                    self.logger.info(f"Latest price: {df['price'].iloc[0]}")
                    self.logger.info(f"Oldest price: {df['price'].iloc[-1]}")
                
                return df
        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None

    async def process_tick(self, tick):
        """
        Handle a single tick of data. This example shows a simple 
        open/close check: if the last minute close > last minute open
        and the current price exceeds the current minute open, buy TSLA.
        """
        # Update last minute open and close prices
        if tick['timestamp'].second == 0:
            self.last_minute_open = tick['price']
        elif tick['timestamp'].second == 59:
            self.last_minute_close = tick['price']

        # Check if last minute close > last minute open and price is above current minute open
        if self.last_minute_close and self.last_minute_open:
            if self.last_minute_close > self.last_minute_open and tick['price'] > self.last_minute_open:
                # Buy 10,000 USD of TSLA
                quantity = 10000 / tick['price']
                await self.place_order('TSLA', quantity, 'BUY')

        print(f"Processing tick at {tick['timestamp']} with price {tick['price']} and volume {tick['volume']}")

    def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria."""
        if ticks_df is None or len(ticks_df) < 2:
            return False

        current_price = float(ticks_df['price'].iloc[0])
        price_60s_ago = float(ticks_df['price'].iloc[-1])
        
        latest_time = ticks_df['timestamp'].iloc[0]
        cutoff_time = latest_time - timedelta(seconds=15)
        
        ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
        if len(ticks_15s_ago) == 0:
            return False
        
        price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
        
        self.logger.info(f"Current price: {current_price}")
        self.logger.info(f"15s ago price: {price_15s_ago}")
        self.logger.info(f"60s ago price: {price_60s_ago}")

        # Long entry conditions
        if (price_60s_ago < current_price and 
            current_price >= price_15s_ago):
            return True
        return False

    def check_trailing_stop(self, current_price):
        """Check if trailing stop has been hit."""
        if self.position is None:
            return False

        if current_price > self.highest_price:
            self.highest_price = current_price

        stop_price = self.highest_price * (1 - self.trailing_stop)
        
        return current_price <= stop_price

    async def execute_trade(self, action, price, timestamp):
        """Execute a trade order with enhanced exit tracking."""
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
            # Convert timestamp to timezone-naive UTC
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sim_bot_trades 
                    (trade_timestamp, symbol, entry_price, trade_type, trade_direction, quantity)
                    VALUES ($1, 'TSLA', $2, 'MARKET', 'LONG', 1)
                    RETURNING id
                """, timestamp, price)
        except Exception as e:
            self.logger.error(f"Error in log_trade_entry: {e}")
            raise

    async def log_exit_signal(self, price, timestamp):
        """Log when exit conditions are first met."""
        try:
            # Convert timestamp to timezone-naive UTC
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sim_bot_trades 
                    SET exit_signal_price = $1,
                        exit_signal_time = $2
                    WHERE id = $3
                """, price, timestamp, self.current_trade_id)
        except Exception as e:
            self.logger.error(f"Error in log_exit_signal: {e}")
            raise

    async def log_trade_exit(self, price, timestamp):
        """Log actual trade exit details."""
        try:
            # Convert timestamp to timezone-naive UTC
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sim_bot_trades 
                    SET actual_exit_price = $1,
                        actual_exit_time = $2,
                        trade_duration = $2 - trade_timestamp,
                        pnl = $1 - entry_price
                    WHERE id = $3
                """, price, timestamp, self.current_trade_id)
        except Exception as e:
            self.logger.error(f"Error in log_trade_exit: {e}")
            raise

    async def run(self):
        """Main bot loop."""
        self.logger.info("Starting TSLA Long Bot...")
        
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
        bot = TSLALongBot2(db_pool, ib_client, '3_bot')
        
        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
