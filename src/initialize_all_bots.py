"""
Initialize All Bots in Database

This script initializes all trading bots (IDs 1-16) in the database by:
1. Creating initial entries in bot_metrics with zero/null values
2. Creating entries in bot_rankings with default rank scores
3. Verifying bot configurations are readable

Run this before starting bots to ensure they work with the ranking system.
"""

import sys

# Check for required packages
required_packages = ['asyncio', 'asyncpg', 'yaml']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("\nERROR: Missing required packages. Please install:")
    for package in missing_packages:
        print(f"  pip install {package}")
    print("\nThen run this script again.")
    sys.exit(1)

import asyncio
import asyncpg
import logging
import os
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_initialization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("initialize_bots")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

# ALL Bot IDs to initialize
ALL_BOT_IDS = list(range(1, 17))  # Bots 1-16

async def get_db_pool():
    """Create and return a database connection pool."""
    return await asyncpg.create_pool(**DB_CONFIG)

async def read_bot_configs(bot_ids):
    """Read YAML configs for the specified bot IDs."""
    bot_configs = {}
    
    # Try multiple possible bot directory locations
    possible_paths = [
        'bots',                                # If running from src/ directory
        os.path.join('src', 'bots'),           # If running from project root
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'bots'))  # Absolute path
    ]
    
    bots_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            bots_dir = path
            logger.info(f"Found bots directory at: {bots_dir}")
            break
    
    if not bots_dir:
        logger.error(f"Could not find bots directory. Tried: {possible_paths}")
        return bot_configs
    
    for filename in os.listdir(bots_dir):
        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            continue
            
        file_path = os.path.join(bots_dir, filename)
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                
                if config and 'bot_id' in config:
                    # Ensure bot_id is an integer
                    bot_id = int(config['bot_id'])
                    
                    if bot_id in bot_ids:
                        bot_configs[bot_id] = {
                            'ticker': config.get('ticker', 'UNKNOWN'),
                            'algo_id': config.get('algo_id', 0),
                            'name': config.get('name', f"Bot {bot_id}")
                        }
                        logger.info(f"Found config for bot {bot_id}: {config.get('name')}")
        except Exception as e:
            logger.error(f"Error reading config file {filename}: {e}")
    
    return bot_configs

async def clear_existing_bot_data(db_pool, bot_ids):
    """Clear existing data for the specified bots to start with a clean slate."""
    try:
        print("\n⚠️ WARNING: This will delete existing metrics and rankings for bots 1-16 ⚠️")
        print("Are you sure you want to proceed? This cannot be undone. (yes/no): ", end="")
        response = input().strip().lower()
        
        if response != 'yes':
            logger.info("Operation cancelled by user")
            return False
            
        async with db_pool.acquire() as conn:
            # Delete from bot_metrics
            metrics_deleted = await conn.execute("""
                DELETE FROM bot_metrics WHERE bot_id = ANY($1)
            """, bot_ids)
            
            # Delete from bot_rankings
            rankings_deleted = await conn.execute("""
                DELETE FROM bot_rankings WHERE bot_id = ANY($1)
            """, bot_ids)
            
            logger.info(f"Cleared existing data for bots {bot_ids}")
            logger.info(f"Metrics deleted: {metrics_deleted}")
            logger.info(f"Rankings deleted: {rankings_deleted}")
            
            return True
    except Exception as e:
        logger.error(f"Error clearing existing bot data: {e}")
        return False

async def initialize_bot_metrics(db_pool, bot_configs):
    """Initialize bot_metrics table with entries for new bots."""
    try:
        async with db_pool.acquire() as conn:
            # First check if the bot_metrics table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_name = 'bot_metrics'
                );
            """)
            
            if not table_exists:
                logger.error("bot_metrics table does not exist! Please create it first.")
                return False
            
            # Initialize each bot with zero/null metrics
            for bot_id, config in bot_configs.items():
                # Check if bot already exists in metrics
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM bot_metrics WHERE bot_id = $1
                    )
                """, bot_id)
                
                if exists:
                    logger.info(f"Bot {bot_id} already exists in bot_metrics")
                    continue
                
                # Insert new record with minimal data
                await conn.execute("""
                    INSERT INTO bot_metrics (
                        bot_id, 
                        ticker, 
                        algo_id, 
                        timestamp,
                        total_trades,
                        avg_win_rate,
                        profit_per_second,
                        total_pnl,
                        last_updated
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                bot_id, 
                config['ticker'], 
                config['algo_id'], 
                datetime.now(),
                0,  # total_trades
                0.0,  # avg_win_rate
                0.0,  # profit_per_second
                0.0,  # total_pnl
                datetime.now()  # last_updated
                )
                
                logger.info(f"Initialized bot {bot_id} ({config['ticker']}) in bot_metrics table")
            
            return True
    except Exception as e:
        logger.error(f"Error initializing bot_metrics: {e}")
        return False

async def initialize_bot_rankings(db_pool, bot_configs):
    """Initialize bot_rankings table with entries for new bots."""
    try:
        async with db_pool.acquire() as conn:
            # Check if the bot_rankings table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_name = 'bot_rankings'
                );
            """)
            
            if not table_exists:
                logger.error("bot_rankings table does not exist! Please create it first.")
                return False
            
            # Check if rank column exists
            rank_column_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'bot_rankings' AND column_name = 'rank'
                );
            """)
            
            # Get the average rank score of existing bots for default value
            avg_rank = await conn.fetchval("""
                SELECT AVG(rank_score) FROM bot_rankings
            """)
            
            # Use a default if no existing bots
            default_rank = avg_rank if avg_rank is not None else 5.0
            
            # Initialize each bot with default rank
            for bot_id, config in bot_configs.items():
                # Check if bot already exists in rankings
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM bot_rankings WHERE bot_id = $1
                    )
                """, bot_id)
                
                if exists:
                    logger.info(f"Bot {bot_id} already exists in bot_rankings")
                    continue
                
                # Insert new record with default rank based on schema
                if rank_column_exists:
                    await conn.execute("""
                        INSERT INTO bot_rankings (
                            bot_id, 
                            rank_score, 
                            rank, 
                            timestamp,
                            is_active
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, 
                    bot_id, 
                    default_rank,  # Start with average rank 
                    0,  # Will be updated by ranking system
                    datetime.now(), 
                    True)  # Start as active
                else:
                    # Use schema without rank column
                    await conn.execute("""
                        INSERT INTO bot_rankings (
                            bot_id, 
                            rank_score, 
                            timestamp,
                            is_active
                        ) VALUES ($1, $2, $3, $4)
                    """, 
                    bot_id, 
                    default_rank,  # Start with average rank 
                    datetime.now(), 
                    True)  # Start as active
                
                logger.info(f"Initialized bot {bot_id} in bot_rankings table")
            
            # Update all ranks to ensure consistency
            if rank_column_exists:
                await conn.execute("""
                    WITH ranked_bots AS (
                        SELECT 
                            bot_id, 
                            rank_score,
                            ROW_NUMBER() OVER (ORDER BY rank_score DESC) as new_rank
                        FROM bot_rankings
                    )
                    UPDATE bot_rankings br
                    SET rank = rb.new_rank
                    FROM ranked_bots rb
                    WHERE br.bot_id = rb.bot_id
                """)
                logger.info("Updated all bot ranks")
            else:
                logger.info("Skipped rank update as the 'rank' column does not exist in the database schema")
            
            return True
    except Exception as e:
        logger.error(f"Error initializing bot_rankings: {e}")
        return False

async def verify_initialization(db_pool, bot_ids):
    """Verify that all bots have been properly initialized."""
    try:
        async with db_pool.acquire() as conn:
            # Check if rank column exists
            rank_column_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'bot_rankings' AND column_name = 'rank'
                );
            """)
            
            # Check bot_metrics
            metrics_results = await conn.fetch("""
                SELECT bot_id, ticker, algo_id 
                FROM bot_metrics 
                WHERE bot_id = ANY($1)
            """, bot_ids)
            
            # Check bot_rankings - adapt query based on schema
            if rank_column_exists:
                rankings_results = await conn.fetch("""
                    SELECT bot_id, rank_score, rank, is_active
                    FROM bot_rankings
                    WHERE bot_id = ANY($1)
                    ORDER BY rank_score DESC
                """, bot_ids)
            else:
                rankings_results = await conn.fetch("""
                    SELECT bot_id, rank_score, is_active
                    FROM bot_rankings
                    WHERE bot_id = ANY($1)
                    ORDER BY rank_score DESC
                """, bot_ids)
            
            logger.info("\n=== Verification Results ===")
            
            # Metrics results
            logger.info("\nBot Metrics Table:")
            if metrics_results:
                for row in metrics_results:
                    logger.info(f"Bot {row['bot_id']}: Ticker={row['ticker']}, Algo={row['algo_id']}")
            else:
                logger.warning("No bots found in metrics table!")
                
            # Rankings results
            logger.info("\nBot Rankings Table:")
            if rankings_results:
                for row in rankings_results:
                    if rank_column_exists:
                        logger.info(f"Bot {row['bot_id']}: Rank={row['rank']}, Score={row['rank_score']:.2f}, Active={row['is_active']}")
                    else:
                        logger.info(f"Bot {row['bot_id']}: Score={row['rank_score']:.2f}, Active={row['is_active']}")
            else:
                logger.warning("No bots found in rankings table!")
            
            logger.info("\n=== Missing Bots ===")
            metrics_bot_ids = [r['bot_id'] for r in metrics_results]
            rankings_bot_ids = [r['bot_id'] for r in rankings_results]
            
            for bot_id in bot_ids:
                if bot_id not in metrics_bot_ids:
                    logger.warning(f"Bot {bot_id} missing from bot_metrics table")
                if bot_id not in rankings_bot_ids:
                    logger.warning(f"Bot {bot_id} missing from bot_rankings table")
            
            return True
    except Exception as e:
        logger.error(f"Error verifying initialization: {e}")
        return False

async def main():
    """Initialize all bots in the database."""
    try:
        print("\n=== Bot Database Initialization Tool ===\n")
        logger.info("Starting initialization for ALL bots (IDs 1-16)")
        
        # Get database connection pool
        try:
            db_pool = await get_db_pool()
        except Exception as e:
            print(f"\n❌ ERROR: Could not connect to database. Details: {e}")
            print("\nPlease check:")
            print("  1. PostgreSQL is running")
            print("  2. Database 'tick_data' exists")
            print("  3. Username/password are correct in the script")
            return
        
        # Clear existing data first
        cleared = await clear_existing_bot_data(db_pool, ALL_BOT_IDS)
        if not cleared:
            logger.info("Initialization cancelled")
            await db_pool.close()
            return
        
        # Read bot configurations
        bot_configs = await read_bot_configs(ALL_BOT_IDS)
        
        if not bot_configs:
            print("\n❌ ERROR: No bot configurations found!")
            print("\nPlease check:")
            print("  1. Your YAML files are in the 'bots' directory")
            print("  2. Each file has a 'bot_id' field with values 1-16")
            print("  3. The YAML files are valid (no syntax errors)")
            await db_pool.close()
            return
            
        logger.info(f"Found {len(bot_configs)} bot configurations")
        
        # Check if we found all expected bots
        missing_bots = set(ALL_BOT_IDS) - set(bot_configs.keys())
        if missing_bots:
            print(f"\n⚠️ WARNING: Could not find configurations for some bots: {sorted(missing_bots)}")
            
            print("\nDo you want to continue with the bots that were found? (yes/no): ", end="")
            response = input().strip().lower()
            if response != 'yes':
                logger.info("Initialization cancelled")
                await db_pool.close()
                return
        
        # Initialize bot metrics
        metrics_success = await initialize_bot_metrics(db_pool, bot_configs)
        
        # Initialize bot rankings
        rankings_success = await initialize_bot_rankings(db_pool, bot_configs)
        
        # Verify initialization
        if metrics_success and rankings_success:
            await verify_initialization(db_pool, list(bot_configs.keys()))
            print("\n✅ Bot initialization completed successfully!")
            print("\nYou can now run your bots with:")
            print("  python run_bots.py --algo_dir bots")
        else:
            print("\n❌ Bot initialization failed. Check the log for details.")
        
        # Close database connection
        await db_pool.close()
        
    except Exception as e:
        logger.error(f"Unexpected error in bot initialization: {e}")
        print(f"\n❌ ERROR: {e}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    asyncio.run(main()) 