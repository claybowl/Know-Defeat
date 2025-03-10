"""
Run Bots Script

This script demonstrates how to use the BotFactory to:
1. Create trading bots from YAML configuration files
2. Start the bots using the bot_ids defined in the YAML files
"""

import asyncio
import argparse
import logging
import os
import sys
import asyncpg
from base_bot import BotFactory

# Add the project root to the Python path
# This ensures modules like 'src.algorithms.momentum_algorithm' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("run_bots")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def main():
    """Main function to run the trading bots."""
    parser = argparse.ArgumentParser(description='Run trading bots from YAML configurations')
    parser.add_argument('--algo_dir', type=str, default='algorithms', help='Directory containing algorithm YAML files')
    parser.add_argument('--algo_file', type=str, help='Specific algorithm file to run (optional)')
    args = parser.parse_args()
    
    # Create database connection pool
    db_pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        # Create bot factory
        bot_factory = BotFactory(db_pool, algorithm_dir=args.algo_dir)
        
        # Get available algorithms
        if args.algo_file:
            if os.path.exists(args.algo_file):
                algorithm_files = [args.algo_file]
            else:
                logger.error(f"Algorithm file not found: {args.algo_file}")
                return
        else:
            algorithm_files = bot_factory.get_available_algorithms()
        
        if not algorithm_files:
            logger.error(f"No algorithm files found in {args.algo_dir}")
            return
        
        logger.info(f"Found {len(algorithm_files)} algorithm files")
        
        # Create and start bots
        for algo_file in algorithm_files:
            logger.info(f"Creating bot using algorithm: {algo_file}")
            
            # Create the bot - bot ID will be loaded from the YAML file
            bot = await bot_factory.create_bot(algo_file)
            
            if bot:
                # Start the bot
                success = await bot_factory.start_bot(bot.bot_id)
                if success:
                    logger.info(f"Successfully started bot {bot.bot_id}")
                else:
                    logger.error(f"Failed to start bot {bot.bot_id}")
            else:
                logger.error(f"Failed to create bot from {algo_file}")
        
        # Keep the script running
        logger.info("All bots started. Running indefinitely...")
        while True:
            await asyncio.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running bots: {e}")
    finally:
        # Close the database pool
        await db_pool.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
