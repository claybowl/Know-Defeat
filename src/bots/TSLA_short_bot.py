import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pandas as pd
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

class TSLAShortBot:
    def __init__(self, db_pool, ib_client, bot_id):
        """Initialize the trading bot with database connection and parameters."""
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = 6  # fixed bot id for TSLA_short_bot
        self.algo_id = 1  # This bot belongs to algo_id 1
        self.position = None
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        self.lowest_price = float('inf')
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = 10000  # $10,000 position size 
        self.recent_prices = []
        self.price_buffer_size = 2

    def setup_logger(self):
        """Configure logging for the bot."""
        logger = logging.getLogger('TSLAShortBot')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

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

    def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria for shorting."""
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

        # Inverse logic for short entries
        if (price_60s_ago > current_price and 
            current_price <= price_15s_ago):
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
        """Execute a trade order with enhanced exit tracking."""
        try:
            if action == "SELL":  # Opening short position
                self.position = 1
                self.entry_price = price
                self.lowest_price = price
                self.logger.info(f"SELL executed at {price}")
                await self.log_trade_entry(price, timestamp)
                
            elif action == "BUY":  # Closing short position
                pnl = (self.entry_price - price) / self.entry_price * 100
                self.logger.info(f"BUY signal at {price}. PnL: {pnl:.2f}%")
                
                await self.log_exit_signal(price, timestamp)
                actual_exit_price = price
                actual_exit_time = timestamp
                await self.log_trade_exit(actual_exit_price, actual_exit_time)
                
                self.position = None
                self.lowest_price = float('inf')
                self.entry_price = None

        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")
            raise

    async def log_trade_entry(self, price, timestamp):
        """Log trade entry to the database."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            # Convert bot_id string to integer (e.g., '4_bot' -> 4)
            numeric_bot_id = self.bot_id

            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO sim_bot_trades 
                    (entry_time, ticker, entry_price, trade_direction,
                     trade_size, trade_status, bot_id, algo_id)
                    VALUES ($1, 'TSLA', $2, 'SHORT', $3, 'open', $4, $5)
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

    async def run(self):
        """Main bot loop."""
        self.logger.info("Starting TSLA Short Bot...")
        
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
                        await self.execute_trade("BUY", current_price, ticks_df['timestamp'].iloc[0])
                else:
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
        bot = TSLAShortBot(db_pool, ib_client, '4')
        
        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
