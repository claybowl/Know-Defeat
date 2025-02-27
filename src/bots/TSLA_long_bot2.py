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
        self.bot_id = 7  # fixed bot id for TSLA_long_bot2
        self.algo_id = 2  # This bot belongs to algo_id 2
        self.position = None
        self.trailing_stop_pct = Decimal('0.002')  # Convert to Decimal
        self.highest_price = Decimal('0')
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = Decimal('10000')  # Convert to Decimal
        self.trailing_stop_price = None
        self.recent_prices = []
        self.price_buffer_size = 2
        self.last_log_time = time.time()
        self.log_interval = 5  # Only log every 5 seconds
        # Add tracking for current price and minute OHLC
        self.current_price = None
        self.last_minute_open = None
        self.last_minute_close = None
        self.last_minute_high = float('-inf')
        self.last_minute_low = float('inf')

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
                
                # Convert prices to Decimal for consistent arithmetic
                df['price'] = df['price'].apply(lambda x: Decimal(str(x)))

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
        """Analyze if price conditions meet entry criteria for a long trade with a threshold.
        
        For a long trade, we require that the current price is at least 0.1% higher than the price 60 seconds ago
        and that the current price is not lower than the price from 15 seconds ago.
        """
        if ticks_df is None or len(ticks_df) < 2:
            return False
        
        current_price = float(ticks_df['price'].iloc[0])
        price_60s_ago = float(ticks_df['price'].iloc[-1])
        
        # Define a threshold of 0.1% (0.001) for upward movement
        threshold = 0.001
        # Calculate the required minimum price based on 60 seconds ago
        required_price = price_60s_ago * (1 + threshold)
        
        # Log key values for debugging
        self.logger.info(f"Current price: {current_price}, Price 60s ago: {price_60s_ago}")

        # Get the price from approximately 15 seconds ago
        latest_time = ticks_df['timestamp'].iloc[0]
        cutoff_time = latest_time - timedelta(seconds=15)
        ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
        if len(ticks_15s_ago) == 0:
            return False
        price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
        self.logger.info(f"Price 15s ago: {price_15s_ago}")
        
        # Entry condition: current price must be at least the required minimum and also not lower than price 15s ago
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
                self.highest_price = Decimal('0')
                self.entry_price = None

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
                     trade_size, trade_status, bot_id, algo_id)
                    VALUES ($1, 'TSLA', $2, 'LONG', $3, 'open', $4, $5)
                    RETURNING trade_id
                """, timestamp, price, self.position_size, self.bot_id, self.algo_id)

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

    async def log_trade_exit(self, exit_price, exit_time):
        """Log trade exit with proper Decimal handling."""
        try:
            # Handle timezone information
            if exit_time.tzinfo is not None:
                exit_time = exit_time.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                # Convert exit_price to Decimal if it isn't already
                exit_price = Decimal(str(exit_price))
                
                # Calculate PnL using Decimal arithmetic
                entry_price = Decimal(str(self.entry_price))
                pnl_value = ((exit_price - entry_price) / entry_price) * Decimal('100')
                
                await conn.execute("""
                    UPDATE sim_bot_trades 
                    SET exit_price = $1, exit_time = $2, trade_pnl = $3
                    WHERE trade_id = $4
                """, exit_price, exit_time, pnl_value, self.current_trade_id)
                
                self.logger.info(f"Trade exit logged - Price: {exit_price}, PnL: {pnl_value:.2f}%")
        except Exception as e:
            self.logger.error(f"Error in log_trade_exit: {e}")
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
        """Main loop with reduced logging."""
        self.logger.info("Starting TSLA Long Bot 2...")
        
        while True:
            try:
                ticks_df = await self.get_latest_ticks()
                if ticks_df is None or len(ticks_df) == 0:
                    self.logger.warning("No tick data available")
                    await asyncio.sleep(1)
                    continue

                current_price = ticks_df['price'].iloc[0]
                
                # Only log price updates every few seconds or on significant changes
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    self.logger.info(f"Current price: {current_price}")
                    self.last_log_time = current_time

                if self.position is not None:
                    if self.check_trailing_stop(current_price):
                        self.logger.info(f"Stop triggered at {current_price}")
                        await self.execute_trade("SELL", current_price, ticks_df['timestamp'].iloc[0])
                else:
                    if self.analyze_price_conditions(ticks_df):
                        self.logger.info(f"Entry signal at {current_price}")
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
        bot = TSLALongBot2(db_pool, ib_client, 6)

        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
