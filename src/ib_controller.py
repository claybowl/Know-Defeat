import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from bots.COIN_long_bot import CoinLongBot
from bots.TSLA_long_bot import TSLALongBot
from bots.COIN_short_bot import CoinShortBot
from bots.TSLA_short_bot import TSLAShortBot
from bots.COIN_long_bot2 import COINLongBot2
from bots.TSLA_long_bot2 import TSLALongBot2    
from bots.COIN_short_bot2 import COINShortBot2
from bots.TSLA_short_bot2 import TSLAShortBot2
from utils.db_utils import (
    execute_db_query,
    fetch_rows,
    create_db_pool,
    execute_query
)
from metrics_calculator import MetricsCalculator
from metrics_updater import MetricsUpdater
from trade_listener import TradeListener
from bot_ranker import BotRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_controller.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Priority tier - most liquid/important symbols
TIER_1_SYMBOLS = [
    'TSLA',  # Tesla
    'COIN',  # Coinbase
    'SPY',   # S&P 500 ETF
    'QQQ',   # Nasdaq ETF
    'AAPL',  # Apple
]

# Secondary tier - add if performance is good with Tier 1
TIER_2_SYMBOLS = [
    'MSFT',  # Microsoft
    'NVDA',  # NVIDIA
    'META'   # Meta
]

# Optional tier - only add if both Tier 1 and 2 are stable
TIER_3_SYMBOLS = [
    'GOOGL',  # Google
    'AMD'     # AMD
]

# Start with Tier 1, then gradually add more if performance is good
SYMBOLS_TO_STREAM = TIER_1_SYMBOLS  # Start with just Tier 1

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

class BotManager:
    def __init__(self):
        self.bots = []
        self.logger = logging.getLogger(__name__)

    def add_bot(self, bot):
        """Add a new bot to the manager"""
        self.bots.append(bot)
        self.logger.info(f"Added new bot: {bot}")

    async def process_tick(self, ticker, price, timestamp):
        """Process tick data through all registered bots"""
        for bot in self.bots:
            await bot.process_tick(ticker, price, timestamp)

class DataIngestionManager:
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.data_queue = Queue()
        self.app = IBDataIngestion(self.data_queue)
        self.logger = logging.getLogger(__name__)
        self.db_pool = None
        self.bot_manager = BotManager()
        self.metrics_calculator = MetricsCalculator(self.db_pool)
        self.metrics_updater = MetricsUpdater(self.db_pool, self.metrics_calculator)
        self.trade_listener = TradeListener(self.db_pool, self.metrics_updater)
        self.bot_ranker = BotRanker(self.db_pool)

        # Add metrics update task to initialization
        asyncio.create_task(self.update_periodic_metrics())

    # async def init_db(self):
    #     """Initialize database connection pool"""
    #     try:
    #         self.db_pool = await asyncpg.create_pool(
    #             user='postgres',
    #             password='Fuckoff25',  # Replace with your actual password
    #             database='tick_data',
    #             host='localhost',
    #             port=5432,
    #             min_size=5,
    #             max_size=20
    #         )
    #         self.logger.info("Database connection pool initialized")
    #     except Exception as e:
    #         self.logger.error(f"Failed to initialize database pool: {e}")
    #         raise

    async def init_db(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                user='clayb',
                password='musicman',  # Replace with your actual password
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
                # Process all available items in the queue
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    await self.store_tick_data(
                        data['ticker'],
                        data['price'],
                        data['volume'],
                        data['timestamp']
                    )

                # Add this line to process the tick through the bots
                await self.bot_manager.process_tick(
                    data['ticker'],
                    data['price'],
                    data['timestamp']
                )

                # Wait a short time before checking queue again
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1)

    async def update_all_bot_metrics(self):
        """Update comprehensive metrics for all active bots"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get active bots from the last 24 hours
                active_bots = await conn.fetch("""
                    SELECT DISTINCT bot_id, ticker 
                    FROM sim_bot_trades 
                    WHERE entry_time >= NOW() - interval '24 hours'
                """)
                
                self.logger.info(f"Found {len(active_bots)} active bots")
                
                for bot in active_bots:
                    try:
                        bot_id = bot['bot_id']
                        ticker = bot['ticker']
                        
                        # Calculate all performance metrics
                        one_hour_perf = await self.metrics_calculator.calculate_one_hour_performance(bot_id, ticker)
                        two_hour_perf = await self.metrics_calculator.calculate_performance_over_period(bot_id, ticker, hours=2)
                        one_day_perf = await self.metrics_calculator.calculate_performance_over_period(bot_id, ticker, hours=24)
                        one_week_perf = await self.metrics_calculator.calculate_performance_over_period(bot_id, ticker, hours=168)  # 7*24
                        one_month_perf = await self.metrics_calculator.calculate_performance_over_period(bot_id, ticker, hours=720)  # 30*24
                        
                        # Calculate trading statistics
                        win_rate = await self.metrics_calculator.calculate_avg_win_rate(bot_id, ticker)
                        drawdown = await self.metrics_calculator.calculate_max_drawdown(bot_id, ticker)
                        sharpe = await self.metrics_calculator.calculate_and_store_sharpe_ratio(bot_id, ticker)
                        win_streaks = await self.metrics_calculator.calculate_and_insert_win_streaks(bot_id, ticker)
                        profit_factor = await self.metrics_calculator.calculate_profit_factor(bot_id, ticker)
                        total_pnl = await self.metrics_calculator.calculate_total_pnl(bot_id, ticker)
                        
                        # Update bot_metrics table with all metrics
                        await conn.execute("""
                            INSERT INTO bot_metrics (
                                bot_id, ticker, timestamp,
                                one_hour_performance, two_hour_performance,
                                one_day_performance, one_week_performance,
                                one_month_performance, avg_win_rate,
                                avg_drawdown, total_pnl,
                                win_streak_2, win_streak_3,
                                win_streak_4, win_streak_5,
                                sharpe_ratio, profit_factor,
                                last_updated
                            ) VALUES ($1, $2, CURRENT_TIMESTAMP, 
                                $3, $4, $5, $6, $7, $8, $9, $10, 
                                $11, $12, $13, $14, $15, $16,
                                CURRENT_TIMESTAMP)
                            ON CONFLICT (bot_id) DO UPDATE SET
                                ticker = EXCLUDED.ticker,
                                timestamp = CURRENT_TIMESTAMP,
                                one_hour_performance = EXCLUDED.one_hour_performance,
                                two_hour_performance = EXCLUDED.two_hour_performance,
                                one_day_performance = EXCLUDED.one_day_performance,
                                one_week_performance = EXCLUDED.one_week_performance,
                                one_month_performance = EXCLUDED.one_month_performance,
                                avg_win_rate = EXCLUDED.avg_win_rate,
                                avg_drawdown = EXCLUDED.avg_drawdown,
                                total_pnl = EXCLUDED.total_pnl,
                                win_streak_2 = EXCLUDED.win_streak_2,
                                win_streak_3 = EXCLUDED.win_streak_3,
                                win_streak_4 = EXCLUDED.win_streak_4,
                                win_streak_5 = EXCLUDED.win_streak_5,
                                sharpe_ratio = EXCLUDED.sharpe_ratio,
                                profit_factor = EXCLUDED.profit_factor,
                                last_updated = CURRENT_TIMESTAMP;
                        """, 
                        bot_id, ticker,
                        one_hour_perf, two_hour_perf, one_day_perf, 
                        one_week_perf, one_month_perf,
                        win_rate, drawdown['max_drawdown'], total_pnl,
                        win_streaks['streak_2'], win_streaks['streak_3'],
                        win_streaks['streak_4'], win_streaks['streak_5'],
                        sharpe, profit_factor)

                        self.logger.info(f"Updated all metrics for bot {bot_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error updating metrics for bot {bot['bot_id']}: {str(e)}")
                        continue  # Continue with next bot if one fails

            self.logger.info("Completed metrics update for all bots")

        except Exception as e:
            self.logger.error(f"Database error in update_all_bot_metrics: {str(e)}")
            raise

    async def update_periodic_metrics(self):
        """Update time-based performance metrics"""
        while True:
            try:
                # Update metrics for all bots
                await self.update_all_bot_metrics()
                self.logger.info("Updated periodic metrics for all bots")
                
                # During testing, update every minute instead of every hour
                await asyncio.sleep(60)  # Changed from 3600 to 60 for testing
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)  # If there's an error, wait a minute before retrying

    async def start(self):
        """Start the data ingestion process"""
        try:
            # Initialize database connection
            await self.init_db()

            # Start the IB client in a separate thread
            self.app.connect('127.0.0.1', 4002, 0)
            api_thread = Thread(target=self.app.run)
            api_thread.daemon = True  # This ensures the thread will exit when the main program does
            api_thread.start()

            # Wait for connection
            await asyncio.sleep(2)

            # Subscribe to market data for all symbols
            for symbol in self.symbols:
                self.app.subscribe_market_data(symbol)

            self.logger.info("Data ingestion system started successfully")

            # Start processing the queue
            await self.process_queue()
            # Initialize and add the bots
            coin_long_bot = CoinLongBot(self.db_pool, self.app, 1)
            tsla_long_bot = TSLALongBot(self.db_pool, self.app, 5)
            coin_short_bot = CoinShortBot(self.db_pool, self.app, 2)
            tsla_short_bot = TSLAShortBot(self.db_pool, self.app, 6)
            
            # New bots with correct IDs
            coin_long_bot2 = COINLongBot2(self.db_pool, self.app, 3)
            tsla_long_bot2 = TSLALongBot2(self.db_pool, self.app, 7)
            coin_short_bot2 = COINShortBot2(self.db_pool, self.app, 4)
            tsla_short_bot2 = TSLAShortBot2(self.db_pool, self.app, 8)

            self.bot_manager.add_bot(coin_long_bot)
            self.bot_manager.add_bot(tsla_long_bot)
            self.bot_manager.add_bot(coin_short_bot)
            self.bot_manager.add_bot(tsla_short_bot)
            self.bot_manager.add_bot(coin_long_bot2)
            self.bot_manager.add_bot(tsla_long_bot2)
            self.bot_manager.add_bot(coin_short_bot2)
            self.bot_manager.add_bot(tsla_short_bot2)

            # Start metrics updater
            asyncio.create_task(self.update_periodic_metrics())

            # Start trade listener and metrics components
            asyncio.create_task(self.trade_listener.listen_for_trade_completion())

        except Exception as e:
            self.logger.error(f"Failed to start data ingestion: {e}")
            raise

    async def stop(self):
        """Stop the data ingestion process"""
        try:
            if self.app:
                self.app.disconnect()
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("Data ingestion system stopped")
        except Exception as e:
            self.logger.error(f"Error stopping data ingestion: {e}")

async def main():
    """Initialize and run the data ingestion manager with Tier 1 symbols."""
    TIER_1_SYMBOLS = [
        'TSLA',  # Tesla
        'COIN',  # Coinbase
        'SPY',   # S&P 500 ETF
        'QQQ',   # Nasdaq ETF
        'AAPL'   # Apple
    ]

    # Create and start data ingestion manager
    manager = DataIngestionManager(TIER_1_SYMBOLS)

    try:
        await manager.start()
    except KeyboardInterrupt:
        await manager.stop()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
