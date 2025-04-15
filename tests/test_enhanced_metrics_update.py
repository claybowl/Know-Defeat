"""
Integration test for enhanced metrics system

This script tests the complete enhanced metrics system by:
1. Adding new metrics columns to the database
2. Calculating metrics using the enhanced calculator
3. Storing metrics with the enhanced updater
4. Verifying all metrics are correctly stored
"""

import asyncio
import asyncpg
import logging
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the enhanced metrics components
from src.metrics_calculator_improvements import EnhancedMetricsCalculator
from src.enhanced_metrics_updater import EnhancedMetricsUpdater

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

async def verify_metrics_columns(conn):
    """Verify that all required metrics columns exist in the database."""
    # Get existing columns
    columns = await conn.fetch("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'bot_metrics'
    """)
    
    column_names = [col['column_name'] for col in columns]
    
    # Define expected columns, especially new ones
    expected_columns = [
        'sortino_ratio', 'calmar_ratio', 'r_multiple', 
        'max_drawdown_duration', 'recovery_factor', 'drawdown_percent',
        'win_streak_6', 'win_streak_7'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in expected_columns if col not in column_names]
    
    if missing_columns:
        logger.warning(f"Missing columns in bot_metrics table: {missing_columns}")
        logger.warning("Run scripts/update_metrics_schema.sh to add these columns")
        return False
        
    logger.info("All required metrics columns exist in the database")
    return True

async def get_test_bots(conn, limit=3):
    """Get a sample of bots to test with."""
    bots = await conn.fetch("""
        SELECT DISTINCT b.bot_id, b.ticker, b.algorithm_type
        FROM sim_bots b
        JOIN sim_bot_trades t ON b.bot_id = t.bot_id
        WHERE t.trade_status = 'closed'
        GROUP BY b.bot_id, b.ticker, b.algorithm_type
        HAVING COUNT(t.trade_id) > 10
        ORDER BY COUNT(t.trade_id) DESC
        LIMIT $1
    """, limit)
    
    return bots

async def test_metrics_update(bot_id, ticker):
    """Test updating metrics for a specific bot."""
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        # Create enhanced metrics calculator and updater
        calculator = EnhancedMetricsCalculator(pool)
        updater = EnhancedMetricsUpdater(pool, calculator)
        
        # Update metrics for the bot
        logger.info(f"Updating metrics for bot {bot_id}, ticker {ticker}")
        success = await updater.update_bot_metrics(bot_id, ticker)
        
        if not success:
            logger.error(f"Failed to update metrics for bot {bot_id}")
            return False
            
        # Verify metrics were stored properly
        async with pool.acquire() as conn:
            metrics = await conn.fetchrow("""
                SELECT * FROM bot_metrics
                WHERE bot_id = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """, bot_id)
            
            if not metrics:
                logger.error(f"No metrics found for bot {bot_id} after update")
                return False
                
            # Check for new metrics
            for new_metric in ['sortino_ratio', 'calmar_ratio', 'r_multiple', 
                              'max_drawdown_duration', 'recovery_factor', 'drawdown_percent']:
                if new_metric not in metrics.keys() or metrics[new_metric] is None:
                    logger.warning(f"Missing or null {new_metric} for bot {bot_id}")
                    
            logger.info(f"Successfully updated and verified metrics for bot {bot_id}")
            
            # Print a sample of metrics
            key_metrics = {
                'bot_id': metrics['bot_id'],
                'ticker': metrics['ticker'],
                'avg_win_rate': metrics['avg_win_rate'],
                'total_pnl': metrics['total_pnl'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'calmar_ratio': metrics['calmar_ratio'],
                'r_multiple': metrics['r_multiple'],
                'max_drawdown': metrics['max_drawdown'],
                'drawdown_percent': metrics['drawdown_percent'],
                'timestamp': metrics['timestamp']
            }
            
            logger.info(f"Key metrics for bot {bot_id}: {key_metrics}")
            return key_metrics
            
    except Exception as e:
        logger.error(f"Error in test_metrics_update for bot {bot_id}: {e}")
        return False
    finally:
        await pool.close()

async def run_test():
    """Run the integration test."""
    print("=== Enhanced Metrics System Integration Test ===")
    print("This test validates the complete enhanced metrics system.")
    
    # Create pool for initial checks
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        async with pool.acquire() as conn:
            # Verify table schema has all required columns
            print("\n1. Verifying database schema...")
            schema_ok = await verify_metrics_columns(conn)
            
            if not schema_ok:
                print("\n⚠️ Database schema is missing some columns.")
                print("Running scripts/update_metrics_schema.sh to add them...")
                
                # We could run this here, but let's ask the user to run it manually
                print("\nPlease run the following command to update the schema:")
                print("./scripts/update_metrics_schema.sh")
                return
            
            # Get test bots
            print("\n2. Finding bots for testing...")
            bots = await get_test_bots(conn)
            
            if not bots:
                print("❌ No suitable bots found for testing.")
                return
                
            print(f"Found {len(bots)} bots for testing:")
            for bot in bots:
                print(f"  - Bot {bot['bot_id']}: {bot['ticker']} ({bot['algorithm_type']})")
        
        # Test metrics update for each bot
        print("\n3. Testing metrics update for each bot...")
        results = []
        
        for bot in bots:
            bot_metrics = await test_metrics_update(bot['bot_id'], bot['ticker'])
            if bot_metrics:
                results.append(bot_metrics)
                
        # Display results
        if results:
            print("\n=== Test Results ===")
            print(f"Successfully updated metrics for {len(results)} bots.")
            
            df = pd.DataFrame(results)
            print("\nMetrics Summary:")
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("\n❌ Failed to update metrics for any bots.")
                
    except Exception as e:
        print(f"\n❌ Error during integration test: {e}")
    finally:
        await pool.close()

if __name__ == "__main__":
    # Check for required packages
    missing_deps = []
    for package in ['asyncpg', 'pandas', 'tabulate']:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        print("❌ Missing dependencies. Please install:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        sys.exit(1)
    
    # Run the test
    asyncio.run(run_test())