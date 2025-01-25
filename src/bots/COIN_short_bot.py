import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from decimal import Decimal

class CoinShortBot:
    def __init__(self, db_connection):
        """Initialize the trading bot with database connection and parameters."""
        self.db = db_connection
        self.position = None
        self.trailing_stop = 0.002  # 0.2%
        self.lowest_price = float('inf')
        self.entry_price = 0
        self.current_trade_id = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Configure logging for the bot."""
        logger = logging.getLogger('CoinShortBot')
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
                    WHERE ticker = 'COIN' 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                )
                SELECT timestamp::timestamp, price::float 
                FROM tick_data 
                WHERE ticker = 'COIN' 
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
        """Process each tick of data."""
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
            if await self.analyze_price_conditions(price):
                await self.execute_trade("SELL", price)

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

        if (price_60s_ago > current_price and 
            current_price <= price_15s_ago):
            return True
        return False

    def check_trailing_stop(self, current_price):
        """Check if trailing stop has been hit for short position."""
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
            if action == "SELL":
                self.position = 1
                self.entry_price = price
                self.lowest_price = price
                self.logger.info(f"SELL executed at {price}")
                self.log_trade("SELL", price)
            elif action == "BUY":
                pnl = (self.entry_price - price) / self.entry_price * 100
                self.logger.info(f"BUY executed at {price}. PnL: {pnl:.2f}%")
                self.log_trade("BUY", price)
                self.position = None
                self.lowest_price = float('inf')
                self.entry_price = 0
        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")

    def log_trade(self, action, price):
        """Log trade to the database."""
        cursor = self.db.cursor()
        try:
            if action == "SELL":
                query = """
                    INSERT INTO sim_bot_trades 
                    (trade_timestamp, symbol, entry_price, trade_type, trade_direction, quantity)
                    VALUES (NOW(), 'COIN', %s, 'MARKET', 'SHORT', 1)
                    RETURNING id, trade_id
                """
                cursor.execute(query, (price,))
                self.current_trade_id = cursor.fetchone()[0]
            else:  # BUY
                query = """
                    UPDATE sim_bot_trades 
                    SET exit_price = %s,
                        trade_duration = NOW() - trade_timestamp,
                        profit_loss = entry_price - %s
                    WHERE id = %s
                """
                cursor.execute(query, (price, price, self.current_trade_id))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Error logging trade to database: {e}")
            self.db.rollback()
        finally:
            cursor.close()

    def run(self):
        """Main bot loop."""
        self.logger.info("Starting COIN Short Bot...")
        
        while True:
            try:
                ticks_df = self.get_latest_ticks()
                if ticks_df is None or len(ticks_df) == 0:
                    time.sleep(1)
                    continue

                current_price = float(ticks_df['price'].iloc[0])

                if self.position is not None:
                    if self.check_trailing_stop(current_price):
                        self.execute_trade("BUY", current_price)
                else:
                    if self.analyze_price_conditions(ticks_df):
                        self.execute_trade("SELL", current_price)

                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    import psycopg2
    
    db_conn = psycopg2.connect(
        dbname="tick_data",
        user="clayb",
        password="musicman",
        host="localhost"
    )
    
    bot = CoinShortBot(db_conn)
    bot.run()
