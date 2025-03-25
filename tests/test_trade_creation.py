"""
Simple test script to create a test trade and update metrics.
This is a more focused test just looking at the trading and metrics part.
"""

import asyncio
import asyncpg
import logging
import sys
import os
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.metrics_calculator import MetricsCalculator
from src.metrics_updater import MetricsUpdater

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

async def create_test_trade():
    """Create a test trade and update metrics."""
    # Create a connection pool
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        # Create metrics calculator and updater
        metrics_calculator = MetricsCalculator(pool)
        metrics_updater = MetricsUpdater(pool, metrics_calculator)
        
        # Find a bot to use for testing
        async with pool.acquire() as conn:
            # Get a bot
            bot = await conn.fetchrow(
                "SELECT bot_id, ticker, algorithm_type FROM sim_bots LIMIT 1"
            )
            
            if not bot:
                logger.error("No bots found in the database. Register at least one bot first.")
                return
            
            bot_id = bot['bot_id']
            ticker = bot['ticker']
            algorithm_type = bot['algorithm_type']
            
            logger.info(f"Using bot {bot_id} with ticker {ticker} for testing")
            
            # Get current price for the ticker
            price_row = await conn.fetchrow(
                "SELECT price FROM tick_data WHERE ticker = $1 ORDER BY timestamp DESC LIMIT 1",
                ticker
            )
            
            if not price_row:
                logger.error(f"No price data found for ticker {ticker}")
                return
                
            current_price = price_row['price']
            
            # Convert current_price to float and generate random prices
            current_price_float = float(current_price)
            entry_price = current_price_float * (1 - random.uniform(0.001, 0.005))
            exit_price = current_price_float * (1 + random.uniform(0.001, 0.005))
            
            # Add test trade
            trade_direction = 'LONG'
            trade_size = 1000.0  # $1000 position size
            
            # Calculate PnL
            if trade_direction == 'LONG':
                pnl = (exit_price - entry_price) * (trade_size / entry_price)
                pnl_percent = ((exit_price / entry_price) - 1) * 100
            else:
                pnl = (entry_price - exit_price) * (trade_size / entry_price)
                pnl_percent = ((entry_price / exit_price) - 1) * 100
                
            # Create trade timestamps
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
                    bot_id, ticker, entry_price, exit_price, trade_size,
                    trade_direction, entry_time, exit_time, 'closed',
                    pnl
                )
            else:
                trade_id = await conn.fetchval(
                    query, 
                    bot_id, ticker, entry_price, exit_price, trade_size,
                    trade_direction, entry_time, exit_time, 'closed',
                    pnl
                )
            
            logger.info(f"Created test trade {trade_id} for bot {bot_id}")
            logger.info(f"Trade details: {trade_direction} {ticker} @ ${entry_price:.2f}, exit @ ${exit_price:.2f}")
            logger.info(f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
            
            # Update metrics for the bot
            result = await metrics_updater.update_bot_metrics(bot_id, ticker)
            if result:
                logger.info(f"Successfully updated metrics for bot {bot_id}")
                
                # Fetch and display updated metrics
                metrics = await conn.fetchrow("""
                    SELECT * FROM bot_metrics 
                    WHERE bot_id = $1 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, bot_id)
                
                if metrics:
                    logger.info("----- Updated Metrics -----")
                    logger.info(f"Total trades: {metrics['total_trades']}")
                    logger.info(f"Win rate: {metrics['avg_win_rate']}%")
                    logger.info(f"Total PnL: ${metrics['total_pnl']}")
                    logger.info(f"1-hour performance: {metrics['one_hour_performance']}")
                    logger.info(f"1-day performance: {metrics['one_day_performance']}")
                    logger.info("--------------------------")
                else:
                    logger.warning("Metrics were updated but could not be retrieved.")
            else:
                logger.error(f"Failed to update metrics for bot {bot_id}")
    
    except Exception as e:
        logger.error(f"Error in test: {e}")
    
    finally:
        # Close the connection pool
        await pool.close()

if __name__ == "__main__":
    asyncio.run(create_test_trade())