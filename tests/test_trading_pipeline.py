"""
Tests the end-to-end trading pipeline from algorithm signal to trade execution
and metrics calculation.
"""

import asyncio
import asyncpg
import logging
import sys
import os
from datetime import datetime, timedelta
import random
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.metrics_calculator import MetricsCalculator
from src.metrics_updater import MetricsUpdater
from src.bot_ranker import BotRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

class TestTradingPipeline:
    """Tests the trading pipeline from algorithm signal to metrics calculation."""

    @classmethod
    async def setup_class(cls):
        """Set up test class - create DB connection pool."""
        cls.pool = await asyncpg.create_pool(**DB_CONFIG)
        cls.metrics_calculator = MetricsCalculator(cls.pool)
        cls.metrics_updater = MetricsUpdater(cls.pool, cls.metrics_calculator)
        cls.bot_ranker = BotRanker(cls.pool)
        
        # Verify our bot is registered
        async with cls.pool.acquire() as conn:
            cls.test_bot = await conn.fetchrow(
                "SELECT bot_id, ticker, algorithm_type, algorithm_module FROM sim_bots LIMIT 1"
            )
            
            if not cls.test_bot:
                logger.error("No bots found in the database. Make sure you've registered at least one bot.")
                raise ValueError("No bots found for testing")
                
            cls.bot_id = cls.test_bot['bot_id']
            cls.ticker = cls.test_bot['ticker']
            cls.algorithm_type = cls.test_bot['algorithm_type']
            cls.algorithm_module = cls.test_bot['algorithm_module']
            
            logger.info(f"Using bot {cls.bot_id} with ticker {cls.ticker} for testing")

    @classmethod
    async def teardown_class(cls):
        """Clean up after tests."""
        await cls.pool.close()

    async def test_01_simulate_trade(self):
        """Test creating a trade in the sim_bot_trades table."""
        # Get current price for the ticker
        async with self.pool.acquire() as conn:
            price_row = await conn.fetchrow(
                "SELECT price FROM tick_data WHERE ticker = $1 ORDER BY timestamp DESC LIMIT 1",
                self.ticker
            )
            
            if not price_row:
                logger.error(f"No price data found for ticker {self.ticker}")
                assert False, f"No price data found for ticker {self.ticker}"
                
            current_price = price_row['price']
            
            # Convert current_price to float and generate random prices
            current_price_float = float(current_price)
            entry_price = current_price_float * (1 - random.uniform(0.001, 0.005))
            exit_price = current_price_float * (1 + random.uniform(0.001, 0.005))
            
            # Add a test trade
            trade_direction = 'LONG'
            trade_size = 1000.0  # $1000 position size
            
            # Calculate PnL
            if trade_direction == 'LONG':
                pnl = (exit_price - entry_price) * (trade_size / entry_price)
                pnl_percent = ((exit_price / entry_price) - 1) * 100
            else:
                pnl = (entry_price - exit_price) * (trade_size / entry_price)
                pnl_percent = ((entry_price / exit_price) - 1) * 100
                
            # Simulate entry time in the past
            entry_time = datetime.utcnow() - timedelta(minutes=30)
            exit_time = datetime.utcnow() - timedelta(minutes=5)
            
            # First, let's check the actual column names in the table
            columns_info = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'sim_bot_trades'
            """)
            
            # Extract column names
            column_names = [col['column_name'] for col in columns_info]
            logger.info(f"Available columns in sim_bot_trades: {column_names}")
            
            # Determine correct column name for PnL
            pnl_column = "trade_pnl" if "trade_pnl" in column_names else "pnl"
            
            # Check for algo_id column
            has_algo_id = "algo_id" in column_names
            
            # Check for exit_reason column
            has_exit_reason = "exit_reason" in column_names
            
            # Create dynamic query based on available columns
            if has_algo_id:
                query = f"""
                    INSERT INTO sim_bot_trades (
                        bot_id, ticker, entry_price, exit_price, trade_size,
                        trade_direction, entry_time, exit_time, trade_status,
                        {pnl_column}, algo_id
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $1)
                    RETURNING trade_id
                """
            else:
                query = f"""
                    INSERT INTO sim_bot_trades (
                        bot_id, ticker, entry_price, exit_price, trade_size,
                        trade_direction, entry_time, exit_time, trade_status,
                        {pnl_column}
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING trade_id
                """
            
            # Insert test trade with appropriate parameters
            if has_algo_id:
                trade_id = await conn.fetchval(
                    query, 
                    self.bot_id, self.ticker, entry_price, exit_price, trade_size,
                    trade_direction, entry_time, exit_time, 'closed',
                    pnl
                )
            else:
                trade_id = await conn.fetchval(
                    query, 
                    self.bot_id, self.ticker, entry_price, exit_price, trade_size,
                    trade_direction, entry_time, exit_time, 'closed',
                    pnl
                )
            
            # Verify trade was created
            assert trade_id is not None, "Failed to create test trade"
            logger.info(f"Created test trade {trade_id} for bot {self.bot_id}")
            
            # Retrieve the trade
            trade = await conn.fetchrow(
                "SELECT * FROM sim_bot_trades WHERE trade_id = $1",
                trade_id
            )
            
            # Verify trade details
            assert trade['bot_id'] == self.bot_id
            assert trade['ticker'] == self.ticker
            assert trade['trade_status'] == 'closed'
            assert trade['pnl'] is not None
            
            # Save trade_id for next test
            self.trade_id = trade_id
            
            # Wait a moment to ensure the trade is fully processed
            await asyncio.sleep(1)
    
    async def test_02_update_metrics(self):
        """Test that metrics are updated after a trade."""
        # Update metrics for the bot
        result = await self.metrics_updater.update_bot_metrics(self.bot_id, self.ticker)
        assert result, "Failed to update metrics"
        
        # Verify metrics were updated
        async with self.pool.acquire() as conn:
            metrics = await conn.fetchrow("""
                SELECT * FROM bot_metrics 
                WHERE bot_id = $1 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, self.bot_id)
            
            # Verify metrics
            assert metrics is not None, "No metrics found for the bot"
            assert metrics['bot_id'] == self.bot_id
            assert metrics['ticker'] == self.ticker
            assert metrics['total_trades'] > 0
            
            logger.info(f"Verified metrics for bot {self.bot_id}")
            logger.info(f"Total trades: {metrics['total_trades']}")
            logger.info(f"Win rate: {metrics['avg_win_rate']}%")
            logger.info(f"Total PnL: ${metrics['total_pnl']}")
    
    async def test_03_rank_bots(self):
        """Test bot ranking system."""
        # Rank all bots
        ranked_bots = await self.bot_ranker.rank_bots()
        assert ranked_bots, "Failed to rank bots"
        
        # Find our test bot in the rankings
        test_bot_ranking = next((bot for bot in ranked_bots if bot['bot_id'] == self.bot_id), None)
        assert test_bot_ranking is not None, f"Test bot {self.bot_id} not found in rankings"
        
        logger.info(f"Bot {self.bot_id} ranking: {test_bot_ranking['rank']} (score: {test_bot_ranking['rank_score']})")
        
        # Check ranking was saved to database
        async with self.pool.acquire() as conn:
            db_ranking = await conn.fetchrow(
                "SELECT * FROM bot_rankings WHERE bot_id = $1",
                self.bot_id
            )
            
            assert db_ranking is not None, "Bot ranking not saved to database"
            assert db_ranking['rank_score'] == test_bot_ranking['rank_score']
    
    async def test_04_fund_allocation(self):
        """Test fund allocation based on rankings."""
        # Get allocation for a sample fund
        allocations = await self.bot_ranker.get_fund_allocation(total_funds=100000)
        assert allocations, "Failed to get fund allocations"
        
        # Test allocation structure
        assert isinstance(allocations, list)
        assert all('bot_id' in alloc and 'allocation_amount' in alloc for alloc in allocations)
        
        # Check total allocations sum to approximately 100%
        total_allocated = sum(alloc['allocation_percentage'] for alloc in allocations)
        assert 99.5 <= total_allocated <= 100.5, f"Total allocation ({total_allocated}%) not close to 100%"
        
        logger.info(f"Fund allocation test passed. Allocated {len(allocations)} bots.")
        
        # Print allocations for our test bot
        test_bot_allocation = next((a for a in allocations if a['bot_id'] == self.bot_id), None)
        if test_bot_allocation:
            logger.info(f"Bot {self.bot_id} allocation: ${test_bot_allocation['allocation_amount']} ({test_bot_allocation['allocation_percentage']}%)")
        else:
            logger.info(f"Bot {self.bot_id} not included in allocations")

def run_test():
    """Run all tests in the class."""
    test = TestTradingPipeline()
    
    async def run_tests():
        await TestTradingPipeline.setup_class()
        
        try:
            await test.test_01_simulate_trade()
            await test.test_02_update_metrics()
            await test.test_03_rank_bots()
            await test.test_04_fund_allocation()
            
            print("\n✅ All tests passed!")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            print(f"\n❌ Test failed: {e}")
            
        finally:
            await TestTradingPipeline.teardown_class()
    
    asyncio.run(run_tests())

if __name__ == "__main__":
    run_test()