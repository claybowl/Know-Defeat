"""
Simplified IB Controller for Data Ingestion

This script connects to Interactive Brokers, subscribes to market data,
and stores the data in a PostgreSQL database without loading any bots.
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
    'SPY',   # S&P 500 ETF
    'QQQ',   # Nasdaq ETF
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
        self.connect('127.0.0.1', 4002, 0)

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
        nasdaq_symbols = ['COIN','TSLA','NVDA','FOUR','CEG','CVNA','VERA','CYTK','ROOT','JANX','LBPH','ARWR','FLYW']
        if symbol in nasdaq_symbols:
            contract.primaryExchange = 'NASDAQ'

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

class DataIngestionManager:
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.data_queue = Queue()
        self.app = IBDataIngestion(self.data_queue)
        self.logger = logging.getLogger(__name__)
        self.db_pool = None
        # Dictionary to store last valid price for each ticker
        self.last_valid_prices = {}

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
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
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
        while True:
            try:
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

            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1)

    async def start(self):
        """Start the data ingestion process"""
        logger.info("Starting data ingestion manager")
        
        # Initialize the database first
        await self.init_db()
        
        # Connect to Interactive Brokers
        try:
            self.app.connect('127.0.0.1', 4002, 0)
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

async def main():
    """Initialize and run the data ingestion manager with Tier 1 symbols."""
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