from ib_insync import IB, Contract, util
import psycopg2
from datetime import datetime
import logging
import queue
from threading import Thread
import pandas as pd

class IBDataCollector:
    def __init__(self, db_config, symbols):
        """
        Initialize the data collector with database configuration and symbols to track.
        
        Parameters:
        db_config (dict): Database connection parameters
        symbols (list): List of stock symbols to track
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize queues for handling data
        self.tick_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
        # Store symbols we want to track
        self.symbols = symbols
        self.active_contracts = {}
        
        # Initialize IB connection
        self.ib = IB()
        
        # Initialize database connection
        try:
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
            self.logger.info("Successfully connected to database")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise

    def connect_to_ib(self):
        """Establish connection to Interactive Brokers TWS"""
        try:
            self.ib.connect('127.0.0.1', 7497, clientId=1)
            self.logger.info("Successfully connected to IB")
        except Exception as e:
            self.logger.error(f"IB connection failed: {e}")
            raise

    def create_contract(self, symbol):
        """
        Create an IB contract object for a given symbol
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract

    def handle_market_data(self, ticker):
        """
        Callback function for processing incoming market data
        """
        try:
            # Create data point with all relevant information
            data_point = {
                'timestamp': datetime.utcnow(),
                'symbol': ticker.contract.symbol,
                'last_price': ticker.last,
                'bid_price': ticker.bid,
                'ask_price': ticker.ask,
                'volume': ticker.volume,
                'bid_size': ticker.bidSize,
                'ask_size': ticker.askSize
            }
            self.tick_queue.put(data_point)
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            self.error_queue.put(e)

    def store_tick_data(self, data):
        """
        Store a single tick data point in the database
        """
        sql = """
        INSERT INTO tick_data (
            timestamp, symbol, last_price, bid_price, ask_price, 
            volume, bid_size, ask_size
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            self.cursor.execute(sql, (
                data['timestamp'],
                data['symbol'],
                data['last_price'],
                data['bid_price'],
                data['ask_price'],
                data['volume'],
                data['bid_size'],
                data['ask_size']
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database insertion failed: {e}")
            self.conn.rollback()
            raise

    def process_data_queue(self):
        """
        Continuously process data from the queue and store in database
        """
        while True:
            try:
                # Get data from queue with timeout
                data = self.tick_queue.get(timeout=1)
                self.store_tick_data(data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
                self.error_queue.put(e)

    def start_data_collection(self):
        """
        Start the data collection process
        """
        try:
            # Connect to IB
            self.connect_to_ib()
            
            # Start queue processing in separate thread
            processing_thread = Thread(
                target=self.process_data_queue, 
                daemon=True
            )
            processing_thread.start()
            
            # Subscribe to market data for each symbol
            for symbol in self.symbols:
                contract = self.create_contract(symbol)
                self.ib.reqMktData(contract, '', False, False)
                self.active_contracts[symbol] = contract
                self.logger.info(f"Subscribed to market data for {symbol}")
            
            # Start IB event loop
            self.ib.run()
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            raise
