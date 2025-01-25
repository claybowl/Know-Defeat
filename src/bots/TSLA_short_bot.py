import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from decimal import Decimal

class TSLAShortBot:
    def __init__(self, db_pool):
        """Initialize the trading bot with database connection and parameters."""
        self.db_pool = db_pool
        self.db = db_pool  # Initialize the db attribute if needed
        self.position = None
        # Convert trailing stop to Decimal for consistent calculations
        self.trailing_stop = Decimal('0.002')  # 0.2%
        self.lowest_price = Decimal('inf')
        self.entry_price = Decimal('0')
        self.current_trade_id = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Configure logging for the bot."""
        logger = logging.getLogger('TSLAShortBot')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_latest_ticks(self):
        """Fetch the last 60 seconds of tick data from the database."""
        cursor = self.db.cursor()
        try:
            query = """
                WITH latest_tick AS (
                    SELECT timestamp 
                    FROM tick_data 
                    WHERE ticker = 'TSLA' 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                )
                SELECT timestamp::timestamp, price::float 
                FROM tick_data 
                WHERE ticker = 'TSLA' 
                AND timestamp >= (SELECT timestamp - INTERVAL '60 seconds' FROM latest_tick)
                ORDER BY timestamp DESC;
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=['timestamp', 'price'])
            self.logger.info(f"Query returned {len(df)} rows")
            if not df.empty:
                self.logger.info(f"Latest price: {df['price'].iloc[0]}")
                self.logger.info(f"Oldest price: {df['price'].iloc[-1]}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None
        finally:
            cursor.close()

    async def process_tick(self, ticker, price, timestamp):
        """Process each tick and decide on trading actions."""
        # Ensure price is a Decimal
        if not isinstance(price, Decimal):
            price = Decimal(str(price))

        self.logger.info(f"Processing tick for {ticker} at {timestamp}: ${price:.2f}")

        # If we have a position, check trailing stop
        if self.position is not None:
            if self.check_trailing_stop(price):
                await self.execute_trade("BUY", price)
        # If no position, check entry conditions
        else:
            if self.analyze_price_conditions(price):
                await self.execute_trade("SELL", price)

    def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria."""
        if ticks_df is None or len(ticks_df) < 2:
            return False

        current_price = float(ticks_df['price'].iloc[0])
        price_60s_ago = float(ticks_df['price'].iloc[-1])
        
        # Get the timestamp from 15 seconds ago
        latest_time = ticks_df['timestamp'].iloc[0]
        cutoff_time = latest_time - timedelta(seconds=15)
        
        # Find price from 15 seconds ago
        ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
        if len(ticks_15s_ago) == 0:
            return False
        
        price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
        
        # Debug logging
        self.logger.info(f"Current price: {current_price}")
        self.logger.info(f"15s ago price: {price_15s_ago}")
        self.logger.info(f"60s ago price: {price_60s_ago}")

        # Check entry conditions
        if (price_60s_ago > current_price and 
            current_price <= price_15s_ago):
            return True
        return False

    def check_trailing_stop(self, current_price):
        """Check if trailing stop has been hit."""
        if self.position is None:
            return False

        if current_price < self.lowest_price:
            self.lowest_price = current_price

        stop_price = self.lowest_price * (1 + self.trailing_stop)
        
        if current_price >= stop_price:
            return True
        return False

    def execute_trade(self, action, price):
        """Execute a trade order."""
        try:
            if action == "BUY":
                self.position = 1
                self.entry_price = price
                self.lowest_price = price
                self.logger.info(f"BUY executed at {price}")
                
                # Log trade to database
                self.log_trade("BUY", price)
                
            elif action == "SELL":
                pnl = (price - self.entry_price) / self.entry_price * 100
                self.logger.info(f"SELL executed at {price}. PnL: {pnl:.2f}%")
                
                # Log trade to database
                self.log_trade("SELL", price)
                
                self.position = None
                self.lowest_price = 0
                self.entry_price = 0
                
        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")

    async def log_trade(self, action, price):
        """Log trade to the database."""
        async with self.db_pool.acquire() as conn:
            try:
                if action == "SELL":
                    query = """
                        INSERT INTO sim_bot_trades 
                        (trade_timestamp, symbol, entry_price, trade_type, trade_direction, quantity)
                        VALUES (NOW(), 'TSLA', $1, 'MARKET', 'SHORT', 1)
                        RETURNING id, trade_id
                    """
                    result = await conn.fetchrow(query, price)
                    self.current_trade_id = result['id']
                else:  # BUY
                    query = """
                        UPDATE sim_bot_trades 
                        SET exit_price = $1,
                            trade_duration = NOW() - trade_timestamp,
                            profit_loss = entry_price - $2
                        WHERE id = $3
                    """
                    await conn.execute(query, price, price, self.current_trade_id)
            except Exception as e:
                self.logger.error(f"Error logging trade to database: {e}")
                raise

    def run(self):
        """Main bot loop."""
        self.logger.info("Starting TSLA Momentum Bot...")
        
        while True:
            try:
                # Get latest tick data
                ticks_df = self.get_latest_ticks()
                if ticks_df is None or len(ticks_df) == 0:
                    time.sleep(1)
                    continue

                current_price = float(ticks_df['price'].iloc[0])

                # If we have a position, check trailing stop
                if self.position is not None:
                    if self.check_trailing_stop(current_price):
                        self.execute_trade("SELL", current_price)
                # If no position, check entry conditions
                else:
                    if self.analyze_price_conditions(ticks_df):
                        self.execute_trade("BUY", current_price)

                time.sleep(1)  # Wait 1 second before next check

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    import psycopg2

    # Connect to your PostgreSQL database
    db_conn = psycopg2.connect(
        dbname="tick_data",
        user="clayb",
        password="musicman",
        host="localhost"
    )

    # Initialize and run the bot
    bot = TSLAShortBot(db_conn)
    bot.run()
