import asyncio
import logging
import asyncpg
import sys
import os

# Add the src directory to the path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from metrics_calculator import MetricsCalculator
from metrics_updater import MetricsUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Database configuration
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def check_bot_ids():
    """Get a list of bot_ids from the database to test with."""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # First, check if sim_bot_trades table exists
        exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = 'sim_bot_trades'
            );
        """)
        
        if not exists:
            logging.error("The sim_bot_trades table does not exist.")
            return []
        
        # Get distinct bot_ids that have trades
        bot_ids = await conn.fetch("""
            SELECT DISTINCT bot_id
            FROM sim_bot_trades
            LIMIT 10;
        """)
        
        # Check if sim_bot_trades has the right columns
        try:
            columns = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'sim_bot_trades';
            """)
            column_names = [col['column_name'] for col in columns]
            logging.info(f"sim_bot_trades columns: {column_names}")
            
            # Check for critical columns
            critical_columns = ['bot_id', 'algo_id', 'trade_pnl']
            missing = [col for col in critical_columns if col not in column_names]
            if missing:
                logging.error(f"sim_bot_trades is missing critical columns: {missing}")
            
            # Check timestamp-related column
            timestamp_candidates = ['timestamp', 'trade_time', 'entry_time', 'created_at', 'date_time']
            timestamp_col = next((col for col in timestamp_candidates if col in column_names), None)
            if timestamp_col:
                logging.info(f"Found timestamp column: {timestamp_col}")
            else:
                logging.error("No timestamp-like column found in sim_bot_trades")
            
        except Exception as e:
            logging.error(f"Error checking sim_bot_trades columns: {e}")
        
        # If no bot_ids found, try a harder-coded list
        if not bot_ids:
            return [1, 2, 3, 4]  # Default test bot IDs
        
        return [row['bot_id'] for row in bot_ids]
    
    except Exception as e:
        logging.error(f"Error checking bot IDs: {e}")
        return [1, 2, 3, 4]  # Default test bot IDs
    finally:
        if 'conn' in locals():
            await conn.close()

async def verify_metrics_calculation(bot_id, ticker="COIN"):
    """Test metrics calculation for a specific bot."""
    try:
        logging.info(f"Testing metrics calculation for bot_id: {bot_id}, ticker: {ticker}")
        
        # Create a DB connection pool
        pool = await asyncpg.create_pool(**DB_CONFIG)
        
        # Create metrics calculator and updater
        metrics_calculator = MetricsCalculator(pool)
        metrics_updater = MetricsUpdater(pool, metrics_calculator)
        
        # Run the metrics update
        success = await metrics_updater.update_bot_metrics(bot_id, ticker)
        
        if success:
            logging.info(f"Successfully updated metrics for bot_id: {bot_id}")
            
            # Check what was stored
            async with pool.acquire() as conn:
                metrics = await conn.fetchrow("""
                    SELECT * FROM bot_metrics 
                    WHERE bot_id = $1
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, bot_id)
            
            if metrics:
                logging.info("Metrics stored in the database:")
                for key, value in metrics.items():
                    if key not in ('timestamp', 'last_updated'):
                        logging.info(f"  {key}: {value}")
                
                # Check for missing metrics
                missing_metrics = []
                expected_metrics = [
                    'one_hour_performance', 'two_hour_performance', 'one_day_performance', 
                    'one_week_performance', 'one_month_performance', 'avg_win_rate', 
                    'profit_per_second', 'total_pnl', 'total_trades', 'avg_profit_per_trade',
                    'avg_drawdown', 'price_model_score', 'volume_model_score', 'price_wall_score'
                ]
                
                for metric in expected_metrics:
                    if metric not in metrics or metrics[metric] is None:
                        missing_metrics.append(metric)
                
                if missing_metrics:
                    logging.warning(f"Missing metrics: {', '.join(missing_metrics)}")
                else:
                    logging.info("All expected metrics are present!")
                
                return True
            else:
                logging.error("No metrics found in database after update.")
                return False
        else:
            logging.error(f"Failed to update metrics for bot_id: {bot_id}")
            return False
        
    except Exception as e:
        logging.error(f"Error in verify_metrics_calculation: {e}")
        return False
    finally:
        if 'pool' in locals():
            await pool.close()

async def main():
    try:
        # Get bot IDs to test with
        bot_ids = await check_bot_ids()
        
        if not bot_ids:
            logging.error("No bot IDs found to test with.")
            return
        
        logging.info(f"Testing with bot IDs: {bot_ids}")
        
        # Test each bot ID
        results = []
        for bot_id in bot_ids:
            result = await verify_metrics_calculation(bot_id)
            results.append((bot_id, result))
        
        # Summarize results
        successes = sum(1 for _, success in results if success)
        logging.info(f"Tests completed. {successes}/{len(results)} successful.")
        
        # Detail any failures
        failures = [(bot_id, success) for bot_id, success in results if not success]
        if failures:
            logging.warning(f"Failed tests: {failures}")
    
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 