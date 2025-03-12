"""
Initialize All Bots in Database

This script initializes all trading bots (IDs 1-126) in the database by:
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

# Configure logging with explicit encoding for file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)

# Add file handler separately with encoding specified
file_handler = logging.FileHandler('bot_initialization.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("initialize_bots")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

# ALL Bot IDs to initialize
ALL_BOT_IDS = list(range(1, 127))  # Bots 1-126

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
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'bots')),  # Absolute path
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bots'))  # One level up
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
    
    # Get a list of all YAML files in the directory
    yaml_files = [f for f in os.listdir(bots_dir) if f.endswith(('.yaml', '.yml'))]
    logger.info(f"Found {len(yaml_files)} YAML files in the bots directory")
    
    # Process all YAML files
    for filename in yaml_files:
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
                        logger.info(f"Found config for bot {bot_id}: {config.get('name', f'Bot {bot_id}')} (Ticker: {config.get('ticker', 'UNKNOWN')}, Algo: {config.get('algo_id', 0)})")
        except Exception as e:
            logger.error(f"Error reading config file {filename}: {e}")
    
    # Log summary of found configs
    found_ids = set(bot_configs.keys())
    missing_ids = set(bot_ids) - found_ids
    
    logger.info(f"Found configurations for {len(bot_configs)} bots")
    if missing_ids:
        logger.warning(f"Missing configurations for {len(missing_ids)} bots: {sorted(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}")
    
    return bot_configs

async def clear_existing_bot_data(db_pool, bot_ids):
    """Clear existing data for the specified bots to start with a clean slate."""
    try:
        print("\n[WARNING] This will delete existing metrics and rankings for bots 1-126 [WARNING]")
        print("Are you sure you want to proceed? This cannot be undone. (yes/no): ", end="")
        response = input().strip().lower()
        
        if response != 'yes':
            logger.info("Operation cancelled by user")
            return False
            
        async with db_pool.acquire() as conn:
            # Count existing records for reporting
            metrics_count = await conn.fetchval("""
                SELECT COUNT(*) FROM bot_metrics WHERE bot_id = ANY($1)
            """, bot_ids)
            
            rankings_count = await conn.fetchval("""
                SELECT COUNT(*) FROM bot_rankings WHERE bot_id = ANY($1)
            """, bot_ids)
            
            logger.info(f"Found {metrics_count} existing metrics records and {rankings_count} existing rankings records")
            
            # Process in batches for better performance with many bots
            batch_size = 50  # Adjust based on performance testing
            total_metrics_deleted = 0
            total_rankings_deleted = 0
            
            for i in range(0, len(bot_ids), batch_size):
                batch = bot_ids[i:i+batch_size]
                
                # Delete from bot_metrics
                metrics_result = await conn.execute("""
                    DELETE FROM bot_metrics WHERE bot_id = ANY($1)
                """, batch)
                
                # Delete from bot_rankings
                rankings_result = await conn.execute("""
                    DELETE FROM bot_rankings WHERE bot_id = ANY($1)
                """, batch)
                
                # Extract numbers from command tags (e.g., "DELETE 5" -> 5)
                try:
                    metrics_deleted = int(metrics_result.split(' ')[1]) if metrics_result else 0
                    rankings_deleted = int(rankings_result.split(' ')[1]) if rankings_result else 0
                    
                    total_metrics_deleted += metrics_deleted
                    total_rankings_deleted += rankings_deleted
                    
                    logger.info(f"Batch {i//batch_size + 1}: Deleted {metrics_deleted} metrics and {rankings_deleted} rankings")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse deletion counts from results: {metrics_result}, {rankings_result}")
            
            logger.info(f"Cleared existing data for bots 1-126")
            logger.info(f"Total metrics deleted: {total_metrics_deleted}")
            logger.info(f"Total rankings deleted: {total_rankings_deleted}")
            
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
            
            # Get existing bot IDs to avoid duplicates
            existing_bot_ids = set(await conn.fetch("SELECT bot_id FROM bot_metrics"))
            existing_bot_ids = {row['bot_id'] for row in existing_bot_ids}
            
            # Prepare batch insert for better performance
            now = datetime.now()
            records_to_insert = []
            
            for bot_id, config in bot_configs.items():
                if bot_id in existing_bot_ids:
                    logger.info(f"Bot {bot_id} already exists in bot_metrics")
                    continue
                
                records_to_insert.append((
                    bot_id, 
                    config['ticker'], 
                    config['algo_id'], 
                    now,  # timestamp
                    0,    # total_trades
                    0.0,  # avg_win_rate
                    0.0,  # profit_per_second
                    0.0,  # total_pnl
                    now   # last_updated
                ))
            
            # Insert in batches to improve performance
            if records_to_insert:
                batch_size = 50  # Adjust based on performance testing
                for i in range(0, len(records_to_insert), batch_size):
                    batch = records_to_insert[i:i+batch_size]
                    await conn.executemany("""
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
                    """, batch)
                    
                logger.info(f"Initialized {len(records_to_insert)} bots in bot_metrics table")
            else:
                logger.info("No new bots to initialize in bot_metrics table")
            
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
            
            # Get existing bot IDs to avoid duplicates
            existing_bot_ids = set(await conn.fetch("SELECT bot_id FROM bot_rankings"))
            existing_bot_ids = {row['bot_id'] for row in existing_bot_ids}
            
            # Prepare batch insert for better performance
            now = datetime.now()
            records_to_insert = []
            
            for bot_id, config in bot_configs.items():
                if bot_id in existing_bot_ids:
                    logger.info(f"Bot {bot_id} already exists in bot_rankings")
                    continue
                
                if rank_column_exists:
                    records_to_insert.append((
                        bot_id,
                        default_rank,  # rank_score
                        0,             # rank (will be updated later)
                        now,           # timestamp
                        True           # is_active
                    ))
                else:
                    records_to_insert.append((
                        bot_id,
                        default_rank,  # rank_score
                        now,           # timestamp
                        True           # is_active
                    ))
            
            # Insert in batches to improve performance
            if records_to_insert:
                batch_size = 50  # Adjust based on performance testing
                for i in range(0, len(records_to_insert), batch_size):
                    batch = records_to_insert[i:i+batch_size]
                    
                    if rank_column_exists:
                        await conn.executemany("""
                            INSERT INTO bot_rankings (
                                bot_id, 
                                rank_score, 
                                rank, 
                                timestamp,
                                is_active
                            ) VALUES ($1, $2, $3, $4, $5)
                        """, batch)
                    else:
                        await conn.executemany("""
                            INSERT INTO bot_rankings (
                                bot_id, 
                                rank_score, 
                                timestamp,
                                is_active
                            ) VALUES ($1, $2, $3, $4)
                        """, batch)
                
                logger.info(f"Initialized {len(records_to_insert)} bots in bot_rankings table")
            else:
                logger.info("No new bots to initialize in bot_rankings table")
            
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
            
            # Count bots in bot_metrics
            metrics_count = await conn.fetchval("""
                SELECT COUNT(*) FROM bot_metrics WHERE bot_id = ANY($1)
            """, bot_ids)
            
            # Count bots in bot_rankings
            rankings_count = await conn.fetchval("""
                SELECT COUNT(*) FROM bot_rankings WHERE bot_id = ANY($1)
            """, bot_ids)
            
            # Get a sample of bots for detailed reporting
            sample_size = min(10, len(bot_ids))
            
            # Get sample of bot_metrics
            metrics_sample = await conn.fetch("""
                SELECT bot_id, ticker, algo_id 
                FROM bot_metrics 
                WHERE bot_id = ANY($1)
                ORDER BY bot_id
                LIMIT $2
            """, bot_ids, sample_size)
            
            # Get sample of bot_rankings - adapt query based on schema
            if rank_column_exists:
                rankings_sample = await conn.fetch("""
                    SELECT bot_id, rank_score, rank, is_active
                    FROM bot_rankings
                    WHERE bot_id = ANY($1)
                    ORDER BY rank_score DESC
                    LIMIT $2
                """, bot_ids, sample_size)
            else:
                rankings_sample = await conn.fetch("""
                    SELECT bot_id, rank_score, is_active
                    FROM bot_rankings
                    WHERE bot_id = ANY($1)
                    ORDER BY rank_score DESC
                    LIMIT $2
                """, bot_ids, sample_size)
            
            # Find missing bots
            metrics_bot_ids = set(await conn.fetch("""
                SELECT bot_id FROM bot_metrics WHERE bot_id = ANY($1)
            """, bot_ids))
            metrics_bot_ids = {row['bot_id'] for row in metrics_bot_ids}
            
            rankings_bot_ids = set(await conn.fetch("""
                SELECT bot_id FROM bot_rankings WHERE bot_id = ANY($1)
            """, bot_ids))
            rankings_bot_ids = {row['bot_id'] for row in rankings_bot_ids}
            
            missing_in_metrics = set(bot_ids) - metrics_bot_ids
            missing_in_rankings = set(bot_ids) - rankings_bot_ids
            
            logger.info("\n=== Verification Results ===")
            
            # Summary counts
            logger.info(f"\nTotal Bots Initialized: {len(bot_ids)}")
            logger.info(f"Bots in Metrics Table: {metrics_count}")
            logger.info(f"Bots in Rankings Table: {rankings_count}")
            
            # Metrics sample
            logger.info("\nBot Metrics Sample:")
            if metrics_sample:
                for row in metrics_sample:
                    logger.info(f"Bot {row['bot_id']}: Ticker={row['ticker']}, Algo={row['algo_id']}")
            else:
                logger.warning("No bots found in metrics table!")
                
            # Rankings sample
            logger.info("\nBot Rankings Sample:")
            if rankings_sample:
                for row in rankings_sample:
                    if rank_column_exists:
                        logger.info(f"Bot {row['bot_id']}: Rank={row['rank']}, Score={row['rank_score']:.2f}, Active={row['is_active']}")
                    else:
                        logger.info(f"Bot {row['bot_id']}: Score={row['rank_score']:.2f}, Active={row['is_active']}")
            else:
                logger.warning("No bots found in rankings table!")
            
            # Missing bots summary
            if missing_in_metrics or missing_in_rankings:
                logger.info("\n=== Missing Bots ===")
                
                if missing_in_metrics:
                    missing_count = len(missing_in_metrics)
                    logger.warning(f"{missing_count} bots missing from bot_metrics table")
                    if missing_count <= 10:
                        logger.warning(f"Missing bot IDs: {sorted(missing_in_metrics)}")
                    else:
                        logger.warning(f"First 10 missing bot IDs: {sorted(list(missing_in_metrics))[:10]}...")
                
                if missing_in_rankings:
                    missing_count = len(missing_in_rankings)
                    logger.warning(f"{missing_count} bots missing from bot_rankings table")
                    if missing_count <= 10:
                        logger.warning(f"Missing bot IDs: {sorted(missing_in_rankings)}")
                    else:
                        logger.warning(f"First 10 missing bot IDs: {sorted(list(missing_in_rankings))[:10]}...")
            else:
                logger.info("\n[SUCCESS] All bots successfully initialized in both tables")
            
            return True
    except Exception as e:
        logger.error(f"Error verifying initialization: {e}")
        return False

async def get_algorithm_info(db_pool):
    """Get a list of available algorithms to check against bot configurations."""
    try:
        async with db_pool.acquire() as conn:
            # Check if algorithms table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_name = 'algorithms'
                );
            """)
            
            if not table_exists:
                logger.info("No algorithms table found. Skipping algorithm verification.")
                return {}
            
            # Get algorithm data
            algorithms = await conn.fetch("""
                SELECT algorithm_id, name, description FROM algorithms
            """)
            
            algo_map = {row['algorithm_id']: row['name'] for row in algorithms}
            
            logger.info(f"Found {len(algo_map)} algorithms in the database")
            return algo_map
    except Exception as e:
        logger.error(f"Error retrieving algorithm information: {e}")
        return {}

async def main():
    """Initialize all bots in the database."""
    try:
        print("\n=== Bot Database Initialization Tool ===\n")
        logger.info("Starting initialization for ALL bots (IDs 1-126)")
        
        # Get database connection pool
        try:
            db_pool = await get_db_pool()
        except Exception as e:
            print(f"\n[ERROR] Could not connect to database. Details: {e}")
            print("\nPlease check:")
            print("  1. PostgreSQL is running")
            print("  2. Database 'tick_data' exists")
            print("  3. Username/password are correct in the script")
            return
        
        # Get algorithm information for verification
        algorithms = await get_algorithm_info(db_pool)
        
        # Clear existing data first
        cleared = await clear_existing_bot_data(db_pool, ALL_BOT_IDS)
        if not cleared:
            logger.info("Initialization cancelled")
            await db_pool.close()
            return
        
        # Read bot configurations
        bot_configs = await read_bot_configs(ALL_BOT_IDS)
        
        if not bot_configs:
            print("\n[ERROR] No bot configurations found!")
            print("\nPlease check:")
            print("  1. Your YAML files are in the 'bots' directory")
            print("  2. Each file has a 'bot_id' field with values 1-126")
            print("  3. The YAML files are valid (no syntax errors)")
            await db_pool.close()
            return
            
        logger.info(f"Found {len(bot_configs)} bot configurations")
        
        # Check if we found all expected bots
        missing_bots = set(ALL_BOT_IDS) - set(bot_configs.keys())
        if missing_bots:
            print(f"\n[WARNING] Could not find configurations for {len(missing_bots)} bots")
            print(f"  First few missing bot IDs: {sorted(list(missing_bots))[:5]}{'...' if len(missing_bots) > 5 else ''}")
            
            print("\nDo you want to continue with the bots that were found? (yes/no): ", end="")
            response = input().strip().lower()
            if response != 'yes':
                logger.info("Initialization cancelled")
                await db_pool.close()
                return
        
        # Verify algorithm IDs if we have algorithm data
        if algorithms:
            invalid_algos = []
            for bot_id, config in bot_configs.items():
                if config['algo_id'] not in algorithms:
                    invalid_algos.append((bot_id, config['algo_id']))
            
            if invalid_algos:
                print(f"\n[WARNING] {len(invalid_algos)} bots reference non-existent algorithm IDs")
                for bot_id, algo_id in invalid_algos[:5]:
                    print(f"  Bot {bot_id} references unknown algorithm ID: {algo_id}")
                if len(invalid_algos) > 5:
                    print("  ... and more")
                
                print("\nDo you want to continue anyway? (yes/no): ", end="")
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
            print("\n[SUCCESS] Bot initialization completed successfully!")
            print(f"\nInitialized {len(bot_configs)} out of {len(ALL_BOT_IDS)} bots")
            print("\nYou can now run your bots with:")
            print("  python run_bots.py --algo_dir bots")
        else:
            print("\n[ERROR] Bot initialization failed. Check the log for details.")
        
        # Close database connection
        await db_pool.close()
        
    except Exception as e:
        logger.error(f"Unexpected error in bot initialization: {e}")
        print(f"\n[ERROR] {e}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    asyncio.run(main())
