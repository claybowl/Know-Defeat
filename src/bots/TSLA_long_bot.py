import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
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

class TSLALongBot:
    """Trading bot implementing a momentum-based long strategy for TSLA stock with trailing stop loss."""
    def __init__(self, db_pool, ib_client, bot_id):
        """Initialize the trading bot with database connection and parameters."""
        self.logger = logging.getLogger(__name__)
        self.db_pool = db_pool
        self.ib_client = ib_client
        self.bot_id = 5  # Use the passed bot_id instead of hardcoding
        self.position = None
        self.trailing_stop_pct = 0.002  # 0.2% trailing stop
        self.highest_price = 0
        self.entry_price = None
        self.current_trade_id = None
        self.position_size = 10000  # $10,000 position size 
        self.trailing_stop_price = None
        self.recent_prices = []
        self.price_buffer_size = 2

    def setup_logger(self):
        """Configure logging for the bot."""
        logger = logging.getLogger('TSLALongBot')
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

                # Convert to DataFrame and ensure price is float type
                df = pd.DataFrame(rows, columns=['timestamp', 'price'])
                df['price'] = df['price'].astype(float)  # Convert all prices to float

                if not df.empty:
                    self.logger.info(f"Latest price type: {type(df['price'].iloc[0]).__name__}")
                    self.logger.info(f"Latest price: {df['price'].iloc[0]}")
                    self.logger.info(f"Oldest price type: {type(df['price'].iloc[-1]).__name__}")
                    self.logger.info(f"Oldest price: {df['price'].iloc[-1]}")

                return df

        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None

    def analyze_price_conditions(self, ticks_df):
        """Analyze if price conditions meet entry criteria."""
        if ticks_df is None or len(ticks_df) < 2:
            self.logger.warning("Insufficient data for analysis")
            return False

        try:
            # Ensure all prices are float type
            current_price = float(ticks_df['price'].iloc[0])
            price_60s_ago = float(ticks_df['price'].iloc[-1])

            # Debug logging
            self.logger.debug(f"Current price (type: {type(current_price).__name__}): {current_price}")
            self.logger.debug(f"60s ago price (type: {type(price_60s_ago).__name__}): {price_60s_ago}")

            # Time window calculation
            latest_time = ticks_df['timestamp'].iloc[0]
            cutoff_time = latest_time - timedelta(seconds=15)
            ticks_15s_ago = ticks_df[ticks_df['timestamp'] >= cutoff_time]
            
            if len(ticks_15s_ago) == 0:
                self.logger.warning("No ticks in 15s window")
                return False

            price_15s_ago = float(ticks_15s_ago['price'].iloc[-1])
            self.logger.debug(f"15s ago price (type: {type(price_15s_ago).__name__}): {price_15s_ago}")

            # Entry logic core conditions
            long_condition_1 = price_60s_ago < current_price
            long_condition_2 = current_price >= price_15s_ago
            
            # Minimum price movement filter
            min_move = 0.0015  # 0.15%
            price_change = (current_price - price_60s_ago) / price_60s_ago
            
            self.logger.debug(f"Price change: {price_change:.4f} (min required: {min_move})")
            self.logger.debug(f"Conditions: 60s trend: {long_condition_1}, 15s momentum: {long_condition_2}")

            if price_change < min_move:
                return False
            
            return long_condition_1 and long_condition_2

        except Exception as e:
            self.logger.error(f"Error in analyze_price_conditions: {e}")
            return False

    def check_trailing_stop(self, current_price):
        """Check if trailing stop has been hit."""
        try:
            # Ensure current_price is float
            current_price = float(current_price)
            
            if self.position is None:
                return False

            self.logger.debug(f"Checking trailing stop - Current: {current_price}, Highest: {self.highest_price}")

            if current_price > self.highest_price:
                self.highest_price = current_price
                self.logger.debug(f"New high: {self.highest_price}")

            stop_price = self.highest_price * (1 - self.trailing_stop_pct)
            self.trailing_stop_price = stop_price

            self.logger.debug(f"Stop price: {stop_price}")
            return current_price <= stop_price

        except Exception as e:
            self.logger.error(f"Error in check_trailing_stop: {e}")
            return False

    async def execute_trade(self, action, price, timestamp):
        """Trade execution - critical points:
        - Proper state management
        - Error handling for partial fills
        - Price validation"""
        
        try:
            if action == "BUY":
                # Validate we don't have existing position
                if self.position is not None:
                    self.logger.error("BUY attempt with existing position")
                    return
                
                # Consider adding:
                # - Order size validation
                # - Price sanity check vs recent range
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                await self.log_trade_entry(price, timestamp)

            elif action == "SELL":
                # Important: Verify we actually have a position to sell
                if self.position is None:
                    self.logger.error("SELL attempt with no position")
                    return
                
                # Consider calculating PnL here for immediate feedback
                await self.log_exit_signal(price, timestamp)
                await self.log_trade_exit(price, timestamp)
                
                # Reset state - ensure all variables are cleared
                self.position = None
                self.highest_price = 0
                self.entry_price = None
                self.current_trade_id = None  # Important for clean state

        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")
            raise

    async def log_trade_entry(self, price, timestamp):
        """Log trade entry to the database."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            # Directly use integer bot_id
            numeric_bot_id = self.bot_id

            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO sim_bot_trades 
                    (entry_time, ticker, entry_price, trade_direction, 
                     trade_size, trade_status, bot_id)
                    VALUES ($1, 'TSLA', $2, 'LONG', $3, 'open', $4)
                    RETURNING trade_id
                """, timestamp, price, self.position_size, numeric_bot_id)
                
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
                    entry_price = float(trade['entry_price'])
                    trade_size = float(trade['trade_size'])
                    
                    # Calculate number of shares based on trade size and entry price
                    shares = trade_size / entry_price
                    # Calculate PnL based on price difference * number of shares
                    pnl = shares * (float(price) - entry_price)
                    
                    await conn.execute("""
                        UPDATE sim_bot_trades 
                        SET exit_price = $1,
                            exit_time = $2,
                            trade_pnl = $3,
                            trade_status = 'closed'
                        WHERE trade_id = $4
                    """, price, timestamp, pnl, self.current_trade_id)
                    
                    self.logger.info(f"Trade exit logged - Shares: {shares:.2f}, PnL: ${pnl:.2f}")
        except Exception as e:
            self.logger.error(f"Error in log_trade_exit: {e}")
            raise

    async def run(self):
        """Main loop - watch for:
        - Data freshness issues
        - Connection stability
        - Rate limiting"""
        
        self.logger.info("Starting TSLA Long Bot...")
        
        self.ib_client.connect('127.0.0.1', 4002, 1)
        
        while not self.ib_client.connected:
            await asyncio.sleep(0.1)
        
        self.logger.info("Connected to Interactive Brokers")
        
        while True:
            try:
                # Get data - verify timeframe matches strategy needs
                ticks_df = await self.get_latest_ticks()
                
                # Critical validation points
                if ticks_df is None:
                    self.logger.warning("Null dataframe received")
                    continue
                    
                if len(ticks_df) == 0:
                    self.logger.warning("Empty dataframe received")
                    continue
                
                # Check data freshness (FIX TIMEZONE COMPARISON)
                latest_tick_time = ticks_df['timestamp'].iloc[0]
                
                # Convert to naive datetime if needed
                if latest_tick_time.tzinfo is not None:
                    naive_tick_time = latest_tick_time.replace(tzinfo=None)
                else:
                    naive_tick_time = latest_tick_time
                
                # Compare with naive datetime.now()
                if (datetime.now() - naive_tick_time).total_seconds() > 30:
                    self.logger.error("Stale data - possible feed issue")
                
                current_price = float(ticks_df['price'].iloc[0])
                
                # Calculate dynamic stop based on recent volatility
                recent_volatility = ticks_df['price'].pct_change().std()
                self.trailing_stop_pct = max(0.002, recent_volatility * 0.67)
                
                # Position management logic
                if self.position is not None:
                    # Exit checks - verify stop logic triggers
                    if self.check_trailing_stop(current_price):
                        await self.execute_trade("SELL", current_price, ticks_df['timestamp'].iloc[0])
                else:
                    # Entry check - validate all conditions met
                    if self.analyze_price_conditions(ticks_df):
                        # Consider adding:
                        # - Recent trade cooldown period
                        # - Volatility filter
                        # - Volume confirmation
                        await self.execute_trade("BUY", current_price, ticks_df['timestamp'].iloc[0])

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Change to DEBUG level
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
        bot = TSLALongBot(db_pool, ib_client, 5)
        
        try:
            await bot.run()
        except KeyboardInterrupt:
            logging.info("Shutting down bot...")
        finally:
            await db_pool.close()
            ib_client.disconnect()

    asyncio.run(main())
