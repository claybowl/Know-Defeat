#!/usr/bin/env python3
"""
This script registers all bots from their YAML files in the src/bots/ directory.
It will register bots with IDs between 18 and 126 in both the bot_metrics and bot_rankings tables.

Usage:
    python register_all_bots.py --db_url <DB_URL> [--start_id <START_ID>] [--end_id <END_ID>]

Example:
    python register_all_bots.py --db_url postgres://clayb:musicman@localhost:5432/tick_data
    python register_all_bots.py --db_url postgres://clayb:musicman@localhost:5432/tick_data --start_id 18 --end_id 126
"""

import asyncio
import asyncpg
import argparse
import os
import yaml
import sys
from datetime import datetime

async def register_single_bot(db_pool, bot_id, ticker, algo_id):
    """
    Register a single bot in both the bot_metrics and bot_rankings tables.
    
    Args:
        db_pool: Database connection pool
        bot_id: Bot ID (integer)
        ticker: Ticker symbol 
        algo_id: Algorithm ID (integer)
    """
    try:
        async with db_pool.acquire() as conn:
            # Check if the bot already exists in bot_metrics
            existing_metrics = await conn.fetchrow("SELECT * FROM bot_metrics WHERE bot_id = $1", bot_id)
            
            # Check if the bot already exists in bot_rankings
            existing_rankings = await conn.fetchrow("SELECT * FROM bot_rankings WHERE bot_id = $1", bot_id)
            
            if existing_metrics:
                print(f"Bot {bot_id} is already registered in bot_metrics table.")
                print(f"Current values: ticker={existing_metrics['ticker']}, algo_id={existing_metrics['algo_id']}")
                
                # Update if requested values are different
                if ticker != existing_metrics['ticker'] or algo_id != existing_metrics['algo_id']:
                    await conn.execute("""
                        UPDATE bot_metrics 
                        SET ticker = $2, algo_id = $3, timestamp = NOW()
                        WHERE bot_id = $1
                        """, bot_id, ticker, algo_id)
                    print(f"Updated bot {bot_id} in bot_metrics with ticker '{ticker}' and algo_id '{algo_id}'.")
            else:
                # Insert initial metrics record with default values
                await conn.execute("""
                    INSERT INTO bot_metrics (
                        bot_id, ticker, algo_id, timestamp,
                        one_hour_performance, two_hour_performance, one_day_performance, 
                        one_week_performance, one_month_performance,
                        avg_win_rate, profit_per_second, total_pnl
                    )
                    VALUES ($1, $2, $3, NOW(), 0, 0, 0, 0, 0, 0, 0, 0)
                    """, bot_id, ticker, algo_id)
                print(f"Registered bot {bot_id} in bot_metrics with ticker '{ticker}' and algo_id '{algo_id}'.")
            
            if existing_rankings:
                print(f"Bot {bot_id} is already registered in bot_rankings table.")
                print(f"Current values: rank_score={existing_rankings['rank_score']}, is_active={existing_rankings['is_active']}")
            else:
                # Insert initial rankings record with default values
                await conn.execute("""
                    INSERT INTO bot_rankings (bot_id, rank_score, rank, timestamp, is_active)
                    VALUES ($1, 50.0, 999, NOW(), true)
                    """, bot_id)
                print(f"Registered bot {bot_id} in bot_rankings with default rank score.")
            
    except Exception as e:
        print(f"Error registering bot {bot_id}: {e}")

async def setup_database_tables(db_pool):
    """Create the necessary database tables if they don't exist."""
    try:
        async with db_pool.acquire() as conn:
            # Create bot_metrics table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_metrics (
                    -- Identifiers
                    bot_id INTEGER,
                    ticker VARCHAR(10),
                    algo_id INTEGER,
                    timestamp TIMESTAMP,
                    
                    -- Performance Periods (percentages)
                    one_hour_performance DECIMAL(6,2),
                    two_hour_performance DECIMAL(6,2),
                    one_day_performance DECIMAL(6,2),
                    one_week_performance DECIMAL(6,2),
                    one_month_performance DECIMAL(6,2),
                    
                    -- Core Metrics
                    avg_win_rate DECIMAL(6,2),
                    profit_per_second DECIMAL(10,4),
                    total_pnl DECIMAL(12,2)
                )
            """)
            
            # Create bot_rankings table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_rankings (
                    ranking_id SERIAL PRIMARY KEY,
                    bot_id INTEGER NOT NULL,
                    rank_score DECIMAL(10,2) NOT NULL,
                    rank INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT true,
                    UNIQUE(bot_id)
                )
            """)
            print("Database tables verified/created successfully.")
    except Exception as e:
        print(f"Error setting up database tables: {e}")
        sys.exit(1)

async def process_yaml_files(db_pool, start_id, end_id):
    """Process all YAML files in the bots directory and register bots with IDs in the specified range."""
    # Try multiple possible bot directory locations
    possible_paths = [
        'bots',                                # If running from scripts/ directory
        os.path.join('src', 'bots'),           # If running from project root
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'bots')),  # From scripts dir to src/bots
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'bots'))  # From project root to src/bots
    ]
    
    bots_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            bots_dir = path
            print(f"Found bots directory at: {bots_dir}")
            break
    
    if not bots_dir:
        print(f"Could not find bots directory. Tried: {possible_paths}")
        sys.exit(1)
    
    # Get a list of all YAML files in the directory
    yaml_files = [f for f in os.listdir(bots_dir) if f.endswith(('.yaml', '.yml'))]
    print(f"Found {len(yaml_files)} YAML files in the bots directory")
    
    registered_count = 0
    skipped_count = 0
    
    # Process all YAML files
    for filename in yaml_files:
        file_path = os.path.join(bots_dir, filename)
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                
                bot_id = config.get('bot_id')
                ticker = config.get('ticker')
                algo_id = config.get('algo_id')
                
                # Skip bots that don't have the required fields
                if not all([bot_id, ticker, algo_id]):
                    print(f"Skipping {filename}: Missing required fields (bot_id, ticker, or algo_id)")
                    skipped_count += 1
                    continue
                
                # Skip bots outside the specified ID range
                if bot_id < start_id or bot_id > end_id:
                    print(f"Skipping bot_id {bot_id} (outside specified range)")
                    skipped_count += 1
                    continue
                
                print(f"Processing {filename}: bot_id={bot_id}, ticker={ticker}, algo_id={algo_id}")
                await register_single_bot(db_pool, bot_id, ticker, algo_id)
                registered_count += 1
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            skipped_count += 1
    
    print(f"\nRegistration complete. Registered {registered_count} bots, skipped {skipped_count} bots.")

async def main(db_url, start_id, end_id):
    """Main function to register all bots."""
    # Create a connection pool
    db_pool = await asyncpg.create_pool(db_url)
    
    if not db_pool:
        print(f"Error: Could not connect to database using URL: {db_url}")
        return
    
    try:
        # Set up database tables
        await setup_database_tables(db_pool)
        
        # Process all YAML files and register bots
        await process_yaml_files(db_pool, start_id, end_id)
        
    finally:
        # Close the connection pool
        await db_pool.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register all bots from YAML files in the bots directory.")
    parser.add_argument("--db_url", required=True, help="Database connection URL (e.g., postgres://clayb:musicman@localhost:5432/tick_data)")
    parser.add_argument("--start_id", type=int, default=18, help="Start bot ID for registration range (default: 18)")
    parser.add_argument("--end_id", type=int, default=126, help="End bot ID for registration range (default: 126)")
    args = parser.parse_args()
    
    asyncio.run(main(args.db_url, args.start_id, args.end_id)) 