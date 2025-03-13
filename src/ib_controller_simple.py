"""
Simplified IB Controller for Data Ingestion

This script connects to Interactive Brokers, subscribes to market data,
and stores the data in a PostgreSQL database, with basic bot support.
"""

import sys
import os
import time
from datetime import datetime
from threading import Thread
import logging
import asyncio
import asyncpg
from queue import Queue
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import TickerId, BarData
import yaml
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_controller_simple.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Priority tier - most liquid/important symbols
TIER_1_SYMBOLS = [
    'TSLA',  # Tesla
    'COIN',  # Coinbase
    'SLV',   # Silver ETF
    'NVDA',   # Nvidia
    'ARWR',   # Arrowhead Pharmaceuticals
    'CYTK',   # Cytokinetics
    'ROOT',   # Root Pharmaceuticals
    'JANX',   # Janus Henderson Global Technology Fund
    'AMZN',   # Amazon
    'FLYW',   # Flywire
    'AAPL'   # Apple
]

class IBDataIngestion(EWrapper, EClient):
    def __init__(self, data_queue):
        EClient.__init__(self, self)
        self.data_queue = data_queue
        self.contract_details = {}
        self.logger = logging.getLogger(__name__)

    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Handle error messages from IB API"""
        self.logger.error(f"Error {errorCode}: {errorString}")
        if errorCode == 1100:  # Connectivity between IB and TWS has been lost
            self.handle_disconnect()

    def connectionClosed(self):
        """Handle connection closure"""
        self.logger.warning("Connection to IB closed")
        self.handle_disconnect()

    def connectAck(self):
        """Called when connection is established"""
        self.logger.info(f"Successfully connected to IB Gateway on port 4002")

    def nextValidId(self, orderId: int):
        """Called when connection is ready for trading"""
        self.logger.info("Connection fully established and ready for trading")

    def handle_disconnect(self):
        """Handle disconnection from IB"""
        self.logger.info("Attempting to reconnect...")
        time.sleep(5)  # Wait before reconnecting
        self.connect('127.0.0.1', 4002, 100)

    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """Handle price tick data"""
        if reqId in self.contract_details:
            ticker = self.contract_details[reqId]['symbol']
            timestamp = datetime.utcnow()

            # Skip invalid prices (negative or zero)
            if price <= 0:
                self.logger.warning(f"Received invalid price {price} for {ticker}, skipping")
                return

            # Create a more detailed mapping of tick types
            tick_type_map = {
                1: "BID",
                2: "ASK",
                4: "LAST",
                6: "HIGH",
                7: "LOW",
                9: "CLOSE",
            }

            tick_type_str = tick_type_map.get(tickType, f"UNKNOWN({tickType})")
            self.logger.info(f"Tick Price - {ticker}: {tick_type_str} = ${price:.2f}")

            # Put the data in the queue for async processing
            self.data_queue.put({
                'type': 'price',
                'ticker': ticker,
                'price': price,
                'volume': 0,
                'timestamp': timestamp
            })

    def subscribe_market_data(self, symbol: str, exchange: str = 'SMART', currency: str = 'USD'):
        """Subscribe to market data for a specific symbol"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = exchange
        contract.currency = currency

        # For NASDAQ stocks, set the primary exchange
        nasdaq_symbols = ['COIN','TSLA','NVDA','CEG','CVNA','VERA','CYTK','ROOT','JANX','AMZN','ARWR','FLYW']
        
        # For ETFs, set the secType to 'ETF'
        etf_symbols = ['SLV']
        
        if symbol in nasdaq_symbols:
            contract.primaryExchange = 'NASDAQ'
            
        if symbol in etf_symbols:
            contract.secType = 'ETF'
            contract.primaryExchange = 'ARCA'  # Most ETFs trade on ARCA

        # Store contract details for reference
        req_id = len(self.contract_details) + 1
        self.contract_details[req_id] = {
            'symbol': symbol,
            'exchange': exchange,
            'currency': currency
        }

        # Request market data
        self.reqMktData(req_id, contract, '', False, False, [])
        self.logger.info(f"Subscribed to market data for {symbol}")

class BotManager:
    def __init__(self):
        self.bots = []
        self.logger = logging.getLogger(__name__)
        self.last_processed_data = {}  # Track last processed data by ticker

    async def load_bots_from_db(self, db_pool):
        """Load active bots from the database"""
        try:
            self.logger.info("Loading active bots from database")
            async with db_pool.acquire() as conn:
                # Query to get active bots
                active_bots = await conn.fetch("""
                    SELECT bot_id, ticker, algorithm_type 
                    FROM sim_bots 
                    WHERE is_active = true
                """)
                
                for bot in active_bots:
                    self.add_bot(bot)
                
                self.logger.info(f"Loaded {len(active_bots)} active bots")
                return len(active_bots)
        except Exception as e:
            self.logger.error(f"Failed to load bots: {e}")
            return 0

    def add_bot(self, bot):
        """Add a new bot to the manager"""
        self.bots.append(bot)
        self.logger.info(f"Added bot: {bot['bot_id']} for {bot['ticker']}")

    async def process_tick(self, ticker, price, timestamp, db_pool):
        """Process tick data through all registered bots"""
        if not self.bots:
            # No bots loaded, try to load them
            bots_loaded = await self.load_bots_from_db(db_pool)
            if not bots_loaded:
                self.logger.warning("No bots loaded, cannot process tick data")
                return

        # Store latest tick data in memory cache
        self.last_processed_data[ticker] = {
            'price': price, 
            'timestamp': timestamp
        }
        
        ticker_bots = [bot for bot in self.bots if bot['ticker'] == ticker]
        if not ticker_bots:
            # No bots for this ticker
            return
            
        self.logger.debug(f"Processing tick data for {ticker} ({len(ticker_bots)} bots): ${price:.2f}")
            
        for bot in ticker_bots:
            try:
                # Store the tick processing in the database for the bot to pick up
                async with db_pool.acquire() as conn:
                    # First check if we already have recent tick data (within last 2 seconds)
                    existing_tick = await conn.fetchrow('''
                        SELECT id FROM bot_tick_data
                        WHERE bot_id = $1 AND ticker = $2 
                        AND timestamp > NOW() - INTERVAL '2 seconds'
                        LIMIT 1
                    ''', bot['bot_id'], ticker)
                    
                    if existing_tick:
                        # Update existing tick if it's very recent
                        await conn.execute('''
                            UPDATE bot_tick_data
                            SET price = $1, timestamp = $2, processed = FALSE
                            WHERE id = $3
                        ''', price, timestamp, existing_tick['id'])
                    else:
                        # Insert new tick data
                        await conn.execute('''
                            INSERT INTO bot_tick_data (bot_id, ticker, price, timestamp)
                            VALUES ($1, $2, $3, $4)
                        ''', bot['bot_id'], ticker, price, timestamp)
                    
                    self.logger.debug(f"Processed tick for bot {bot['bot_id']}: {ticker} @ ${price:.2f}")
            except Exception as e:
                self.logger.error(f"Error processing tick for bot {bot['bot_id']}: {e}")

    async def ensure_data_for_all_bots(self, db_pool):
        """
        Ensure all bots have recent tick data by copying data between bots if needed.
        This helps bots that aren't receiving direct ticks for their ticker.
        """
        if not self.last_processed_data:
            return  # No data to propagate
            
        try:
            # Get unique tickers that need data
            bot_tickers = set(bot['ticker'] for bot in self.bots)
            missing_tickers = bot_tickers - set(self.last_processed_data.keys())
            
            if not missing_tickers:
                return  # All tickers have data
                
            self.logger.info(f"Attempting to find data for missing tickers: {missing_tickers}")
            
            async with db_pool.acquire() as conn:
                for ticker in missing_tickers:
                    # Try to find the most recent data for this ticker in tick_data table
                    recent_tick = await conn.fetchrow('''
                        SELECT price, timestamp FROM tick_data
                        WHERE ticker = $1
                        ORDER BY timestamp DESC
                        LIMIT 1
                    ''', ticker)
                    
                    if recent_tick:
                        # We found data, populate it for all bots with this ticker
                        self.logger.info(f"Found recent data for {ticker}: ${recent_tick['price']:.2f}")
                        
                        bots_for_ticker = [bot for bot in self.bots if bot['ticker'] == ticker]
                        for bot in bots_for_ticker:
                            await conn.execute('''
                                INSERT INTO bot_tick_data (bot_id, ticker, price, timestamp)
                                VALUES ($1, $2, $3, $4)
                                ON CONFLICT (bot_id, ticker, timestamp) DO NOTHING
                            ''', bot['bot_id'], ticker, recent_tick['price'], recent_tick['timestamp'])
                            
                        # Update our cache
                        self.last_processed_data[ticker] = {
                            'price': recent_tick['price'],
                            'timestamp': recent_tick['timestamp']
                        }
                    else:
                        self.logger.warning(f"No recent data found for ticker {ticker}")
                        
        except Exception as e:
            self.logger.error(f"Error ensuring data for all bots: {e}")

class DataIngestionManager:
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.data_queue = Queue()
        self.app = IBDataIngestion(self.data_queue)
        self.logger = logging.getLogger(__name__)
        self.db_pool = None
        # Dictionary to store last valid price for each ticker
        self.last_valid_prices = {}
        self.trade_locks = {}  # Dictionary to track locks by trade_id
        self.bot_manager = BotManager()  # Add bot manager

    async def init_db(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                user='clayb',
                password='musicman',
                database='tick_data',
                host='localhost',
                port=5432,
                min_size=5,
                max_size=20
            )
            self.logger.info("Database connection pool initialized")
            
            # Ensure database schema is set up
            await self.ensure_database_schema()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def ensure_database_schema(self):
        """Ensure all required database tables exist"""
        try:
            self.logger.info("Checking database schema...")
            async with self.db_pool.acquire() as conn:
                # Check if sim_bots table exists
                bot_table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sim_bots'
                    )
                """)
                
                if not bot_table_exists:
                    self.logger.info("Creating sim_bots table...")
                    await conn.execute("""
                        CREATE TABLE sim_bots (
                            bot_id INTEGER PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            ticker VARCHAR(10) NOT NULL,
                            algorithm_module VARCHAR(255) NOT NULL,
                            algorithm_type VARCHAR(50) NOT NULL,
                            trade_direction VARCHAR(10) NOT NULL,
                            position_size NUMERIC(15,2) NOT NULL,
                            trailing_stop_pct NUMERIC(8,6) NOT NULL,
                            description TEXT,
                            version VARCHAR(20),
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT NOW(),
                            last_updated TIMESTAMP DEFAULT NOW()
                        )
                    """)
                
                # Check if sim_bot_trades table exists
                trades_table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sim_bot_trades'
                    )
                """)
                
                if not trades_table_exists:
                    self.logger.info("Creating sim_bot_trades table...")
                    await conn.execute("""
                        CREATE TABLE sim_bot_trades (
                            trade_id SERIAL PRIMARY KEY,
                            bot_id INTEGER NOT NULL REFERENCES sim_bots(bot_id),
                            ticker VARCHAR(10) NOT NULL,
                            entry_price NUMERIC(15,6) NOT NULL,
                            exit_price NUMERIC(15,6),
                            trade_size NUMERIC(15,2) NOT NULL,
                            trade_direction VARCHAR(10) NOT NULL,
                            entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
                            exit_time TIMESTAMP,
                            trade_status VARCHAR(20) NOT NULL DEFAULT 'open',
                            pnl NUMERIC(15,2),
                            pnl_percent NUMERIC(15,6),
                            trailing_stop_price NUMERIC(15,6),
                            exit_reason VARCHAR(50)
                        )
                    """)
                
                # Check if bot_tick_data table exists
                tick_table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'bot_tick_data'
                    )
                """)
                
                if not tick_table_exists:
                    self.logger.info("Creating bot_tick_data table...")
                    await conn.execute("""
                        CREATE TABLE bot_tick_data (
                            id SERIAL PRIMARY KEY,
                            bot_id INTEGER NOT NULL,
                            ticker VARCHAR(10) NOT NULL,
                            price NUMERIC(15,6) NOT NULL,
                            timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                            processed BOOLEAN DEFAULT FALSE,
                            CONSTRAINT fk_bot_id
                                FOREIGN KEY(bot_id) 
                                REFERENCES sim_bots(bot_id)
                        )
                    """)
                    
                    # Create index on ticker for faster lookups
                    await conn.execute("""
                        CREATE INDEX idx_bot_tick_data_ticker ON bot_tick_data(ticker)
                    """)
                    
                    # Create index on bot_id for faster lookups
                    await conn.execute("""
                        CREATE INDEX idx_bot_tick_data_bot_id ON bot_tick_data(bot_id)
                    """)
                    
                    # Create index on processed status for faster lookups
                    await conn.execute("""
                        CREATE INDEX idx_bot_tick_data_processed ON bot_tick_data(processed)
                    """)
                
                self.logger.info("Database schema check completed")
                
        except Exception as e:
            self.logger.error(f"Error ensuring database schema: {e}")
            raise

    async def store_tick_data(self, ticker: str, price: float, volume: int, timestamp: datetime):
        """Store tick data in PostgreSQL database"""
        if not self.db_pool:
            self.logger.error("Database pool not initialized")
            return

        try:
            # If price is invalid, use last valid price if available
            if price <= 0:
                if ticker in self.last_valid_prices:
                    self.logger.warning(f"Replacing invalid price {price} with last valid price {self.last_valid_prices[ticker]} for {ticker}")
                    price = self.last_valid_prices[ticker]
                else:
                    self.logger.warning(f"Skipping invalid price {price} for {ticker} (no valid price history)")
                    return  # Skip storing invalid prices if no history
            else:
                # Store valid price for future reference
                self.last_valid_prices[ticker] = price
                
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO tick_data (ticker, price, volume, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', ticker, price, volume, timestamp)
        except Exception as e:
            self.logger.error(f"Failed to store tick data: {e}")

    async def process_queue(self):
        """Process data from the queue and store in database"""
        last_ensure_data_time = time.time()  
        last_check_symbols_time = time.time()
        missing_symbol_count = {}  # Track symbols that aren't getting data
        
        while True:
            try:
                # Periodically ensure all bots have data (every 5 seconds)
                current_time = time.time()
                if current_time - last_ensure_data_time > 5:
                    await self.bot_manager.ensure_data_for_all_bots(self.db_pool)
                    last_ensure_data_time = current_time
                
                # Check if we're getting data for all subscribed symbols (every 30 seconds)
                if current_time - last_check_symbols_time > 30:
                    # Get the set of symbols we're supposed to be tracking
                    received_symbols = set(self.last_valid_prices.keys())
                    expected_symbols = set(self.symbols)
                    
                    missing_symbols = expected_symbols - received_symbols
                    if missing_symbols:
                        # Initialize counter for missing symbols
                        for symbol in missing_symbols:
                            if symbol not in missing_symbol_count:
                                missing_symbol_count[symbol] = 0
                            missing_symbol_count[symbol] += 1
                            
                            # Log warning after we've missed data multiple times
                            if missing_symbol_count[symbol] >= 3:
                                self.logger.warning(f"Not receiving data for {symbol} - checked {missing_symbol_count[symbol]} times")
                                
                                # After several warnings, try to re-subscribe
                                if missing_symbol_count[symbol] >= 5 and missing_symbol_count[symbol] % 5 == 0:
                                    self.logger.info(f"Attempting to re-subscribe to {symbol}")
                                    self.app.subscribe_market_data(symbol)
                    
                    last_check_symbols_time = current_time
                
                # Check if queue is empty before trying to get data
                if self.data_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                # Get data from queue
                data = self.data_queue.get()
                
                # Skip invalid prices
                if data['price'] <= 0:
                    self.logger.warning(f"Skipping invalid price {data['price']} for {data['ticker']} in queue processing")
                    continue
                    
                await self.store_tick_data(
                    data['ticker'],
                    data['price'],
                    data['volume'],
                    data['timestamp']
                )

                # Process tick through bot manager
                await self.bot_manager.process_tick(
                    data['ticker'],
                    data['price'],
                    data['timestamp'],
                    self.db_pool
                )
                
                # Reset the missing count for this symbol since we received data
                if data['ticker'] in missing_symbol_count:
                    missing_symbol_count[data['ticker']] = 0

            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1)

    async def acquire_trade_lock(self, trade_id):
        """Acquire a lock for a specific trade to prevent race conditions"""
        if trade_id not in self.trade_locks:
            self.trade_locks[trade_id] = asyncio.Lock()
        
        await self.trade_locks[trade_id].acquire()
        return True
    
    def release_trade_lock(self, trade_id):
        """Release a trade lock"""
        if trade_id in self.trade_locks and not self.trade_locks[trade_id].locked():
            self.trade_locks[trade_id].release()

    async def complete_trade(self, trade_id, exit_price):
        """Complete an existing trade."""
        try:
            # Acquire lock for this trade
            await self.acquire_trade_lock(trade_id)
            
            async with self.db_pool.acquire() as connection:
                # Get trade details
                trade = await connection.fetchrow("""
                    SELECT bot_id, ticker, entry_price, trade_direction, trade_size
                    FROM sim_bot_trades
                    WHERE trade_id = $1 AND trade_status = 'open'
                """, trade_id)
                
                if not trade:
                    self.logger.error(f"Trade {trade_id} not found or not open")
                    return {
                        'success': False,
                        'reason': 'Trade not found or not open'
                    }
                
                # Calculate P&L based on trade direction
                bot_id = trade['bot_id']
                ticker = trade['ticker']
                entry_price = trade['entry_price']
                direction = trade['trade_direction']
                size = trade['trade_size']
                
                if direction == 'long':
                    pnl = (exit_price - entry_price) * size
                    pnl_percent = ((exit_price / entry_price) - 1) * 100
                else:  # short
                    pnl = (entry_price - exit_price) * size
                    pnl_percent = ((entry_price / exit_price) - 1) * 100
                
                # Update trade record
                await connection.execute("""
                    UPDATE sim_bot_trades
                    SET exit_price = $1, 
                        exit_time = NOW(),
                        trade_status = 'closed',
                        pnl = $2,
                        pnl_percent = $3
                    WHERE trade_id = $4
                """, exit_price, pnl, pnl_percent, trade_id)
                
                self.logger.info(f"Closed trade {trade_id} for bot {bot_id} with PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'bot_id': bot_id,
                    'ticker': ticker,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent
                }
                    
        except Exception as e:
            self.logger.error(f"Error completing trade {trade_id}: {e}")
            return {
                'success': False,
                'reason': f'Exception: {str(e)}'
            }
        finally:
            # Always release the lock
            self.release_trade_lock(trade_id)

    async def register_bots_from_directory(self, directory_path="src/bots"):
        """
        Register all bot YAML definitions from a directory to the database
        """
        try:
            self.logger.info(f"Registering bots from directory: {directory_path}")
            
            # Make sure we have a DB connection
            if not self.db_pool:
                await self.init_db()
                
            # Find all YAML files
            yaml_files = glob.glob(os.path.join(directory_path, "*.yaml"))
            self.logger.info(f"Found {len(yaml_files)} YAML bot definition files")
            
            registered_count = 0
            skipped_count = 0
            
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r') as file:
                        bot_data = yaml.safe_load(file)
                        
                    # Extract required fields
                    bot_id = bot_data.get('bot_id')
                    name = bot_data.get('name')
                    ticker = bot_data.get('ticker')
                    algorithm_module = bot_data.get('algorithm_module')
                    trade_direction = bot_data.get('trade_direction', 'BOTH')  # Default to BOTH if not specified
                    position_size = bot_data.get('position_size', 1000.0)  # Default position size
                    trailing_stop_pct = bot_data.get('trailing_stop_pct', 0.01)  # Default 1% trailing stop
                    description = bot_data.get('description', '')
                    version = str(bot_data.get('version', '1.0'))  # Convert version to string
                    
                    # Get algorithm type from file name
                    file_name = os.path.basename(yaml_file)
                    if '_breakout' in file_name:
                        algorithm_type = 'breakout'
                    elif '_mean_reversion' in file_name:
                        algorithm_type = 'mean_reversion'
                    elif '_price_pattern' in file_name:
                        algorithm_type = 'price_pattern'
                    elif '_support_resistance' in file_name:
                        algorithm_type = 'support_resistance'
                    elif '_volatility_breakout' in file_name:
                        algorithm_type = 'volatility_breakout'
                    else:
                        algorithm_type = 'custom'
                    
                    # Skip if missing required fields
                    if not all([bot_id, ticker, algorithm_module]):
                        self.logger.warning(f"Skipping {yaml_file} - missing required fields")
                        skipped_count += 1
                        continue
                    
                    # Check if bot already exists in database
                    async with self.db_pool.acquire() as conn:
                        existing_bot = await conn.fetchrow(
                            "SELECT bot_id FROM sim_bots WHERE bot_id = $1", 
                            bot_id
                        )
                        
                        if existing_bot:
                            # Update existing bot
                            await conn.execute("""
                                UPDATE sim_bots SET 
                                    name = $1,
                                    ticker = $2,
                                    algorithm_module = $3,
                                    algorithm_type = $4,
                                    trade_direction = $5,
                                    position_size = $6,
                                    trailing_stop_pct = $7,
                                    description = $8,
                                    version = $9,
                                    last_updated = NOW()
                                WHERE bot_id = $10
                            """, name, ticker, algorithm_module, algorithm_type, 
                                trade_direction, position_size, trailing_stop_pct,
                                description, version, bot_id)
                            self.logger.info(f"Updated bot {bot_id} ({name}) in database")
                        else:
                            # Insert new bot
                            await conn.execute("""
                                INSERT INTO sim_bots (
                                    bot_id, name, ticker, algorithm_module, algorithm_type,
                                    trade_direction, position_size, trailing_stop_pct,
                                    description, version, is_active, created_at, last_updated
                                ) VALUES (
                                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TRUE, NOW(), NOW()
                                )
                            """, bot_id, name, ticker, algorithm_module, algorithm_type,
                                trade_direction, position_size, trailing_stop_pct,
                                description, version)
                            self.logger.info(f"Registered new bot {bot_id} ({name}) in database")
                        
                        registered_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing bot file {yaml_file}: {str(e)}")
                    skipped_count += 1
            
            self.logger.info(f"Bot registration completed: {registered_count} registered, {skipped_count} skipped")
            return {
                "success": True,
                "registered": registered_count,
                "skipped": skipped_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to register bots: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def start(self):
        """Start the data ingestion process"""
        logger.info("Starting data ingestion manager")
        
        # Initialize the database first
        await self.init_db()
        
        # Register bots from directory (uncomment to enable during startup)
        # await self.register_bots_from_directory()
        
        # Connect to Interactive Brokers
        try:
            self.app.connect('127.0.0.1', 4002, 100)
            # Start IB client in a separate thread
            api_thread = Thread(target=self.app.run)
            api_thread.daemon = True
            api_thread.start()
            logger.info("Connected to Interactive Brokers")
            
            # Wait for connection to establish
            await asyncio.sleep(2)
            
            # Subscribe to market data for all symbols
            for symbol in self.symbols:
                self.app.subscribe_market_data(symbol)
                logger.info(f"Subscribed to market data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to connect to Interactive Brokers: {e}")
        
        # Load bots from the database
        await self.bot_manager.load_bots_from_db(self.db_pool)
        
        # Start the queue processing task
        self.queue_task = asyncio.create_task(self.process_queue())
        
        logger.info("Data ingestion manager started successfully")

    async def stop(self):
        """Stop the data ingestion process"""
        logger.info("Stopping data ingestion manager")
        
        # Cancel the queue processing task
        if hasattr(self, 'queue_task'):
            self.queue_task.cancel()
            
        # Disconnect from Interactive Brokers
        if hasattr(self, 'app'):
            self.app.disconnect()
            
        # Close the database pool
        if hasattr(self, 'db_pool') and self.db_pool:
            await self.db_pool.close()
            
        logger.info("Data ingestion manager stopped")

# Add a register bots command function to main
async def register_bots():
    """Register all bots from the src/bots directory."""
    logger.info("Starting bot registration process")
    manager = DataIngestionManager([])  # Empty symbols list since we're just registering bots
    
    try:
        # Initialize the database
        await manager.init_db()
        
        # Register bots
        result = await manager.register_bots_from_directory()
        
        if result["success"]:
            logger.info(f"Successfully registered {result['registered']} bots")
        else:
            logger.error(f"Failed to register bots: {result.get('error', 'Unknown error')}")
    finally:
        # Close database connection
        if hasattr(manager, 'db_pool') and manager.db_pool:
            await manager.db_pool.close()
    
    logger.info("Bot registration process completed")

async def main():
    """Initialize and run the data ingestion manager with Tier 1 symbols."""
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "register_bots":
        # Just register the bots and exit
        await register_bots()
        return
        
    # Normal operation - start the data ingestion manager
    manager = DataIngestionManager(TIER_1_SYMBOLS)

    try:
        await manager.start()
        # Keep running indefinitely
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        await manager.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main()) 