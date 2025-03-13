#!/usr/bin/env python3
"""
This script registers a bot in the database by creating entries in both bot_metrics and bot_rankings tables.
If you see the error 'No tick data available', it may be because your bot is not registered in these tables.

Usage:
    python register_bot.py --db_url <DB_URL> --bot_id <BOT_ID> [--ticker <TICKER>] --algo_id <ALGO_ID>

Example:
    python register_bot.py --db_url postgres://clayb:musicman@localhost:5432/tick_data --bot_id 17 --ticker TSLA --algo_id 1
"""

import asyncio
# pylint: disable=import-error
import asyncpg
import argparse
import re
from datetime import datetime

def guess_ticker_from_bot_id(bot_id):
    """
    Try to guess the ticker from the bot_id if it follows a pattern like '17-126'
    where the first part might be a ticker code.
    
    Returns 'UNKNOWN' if no guess can be made.
    """
    # Common tickers in the system
    common_tickers = ['TSLA', 'COIN', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']
    
    # If bot_id contains a dash, try to extract the first part
    if isinstance(bot_id, str) and '-' in bot_id:
        parts = bot_id.split('-')
        if len(parts) >= 2:
            # Try to match the first part with a common ticker
            for ticker in common_tickers:
                if ticker.lower() in parts[0].lower():
                    return ticker
    
    # If no match found, return UNKNOWN
    return 'UNKNOWN'

async def register_bot(db_url, bot_id, ticker, algo_id):
    try:
        # Convert bot_id to integer if it's a string with a dash (like "17-126")
        numeric_bot_id = bot_id
        if isinstance(bot_id, str) and '-' in bot_id:
            # Extract the first number before the dash
            numeric_bot_id = int(bot_id.split('-')[0])
            print(f"Converting bot_id '{bot_id}' to numeric ID: {numeric_bot_id}")
        elif isinstance(bot_id, str) and bot_id.isdigit():
            numeric_bot_id = int(bot_id)
        
        conn = await asyncpg.connect(db_url)
        
        # First, check if the bot_metrics table exists, create if not
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
        
        # Check if the bot_rankings table exists, create if not
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
        
        # Check if the bot already exists in bot_metrics
        existing_metrics = await conn.fetchrow("SELECT * FROM bot_metrics WHERE bot_id = $1", numeric_bot_id)
        
        # Check if the bot already exists in bot_rankings
        existing_rankings = await conn.fetchrow("SELECT * FROM bot_rankings WHERE bot_id = $1", numeric_bot_id)
        
        if existing_metrics:
            print(f"Bot {numeric_bot_id} is already registered in bot_metrics table.")
            print(f"Current values: ticker={existing_metrics['ticker']}, algo_id={existing_metrics['algo_id']}")
            
            # Update if requested values are different
            if ticker != existing_metrics['ticker'] or algo_id != existing_metrics['algo_id']:
                await conn.execute("""
                    UPDATE bot_metrics 
                    SET ticker = $2, algo_id = $3, timestamp = NOW()
                    WHERE bot_id = $1
                    """, numeric_bot_id, ticker, algo_id)
                print(f"Updated bot {numeric_bot_id} in bot_metrics with ticker '{ticker}' and algo_id '{algo_id}'.")
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
                """, numeric_bot_id, ticker, algo_id)
            print(f"Successfully registered bot {numeric_bot_id} in bot_metrics with ticker '{ticker}' and algo_id '{algo_id}'.")
        
        if existing_rankings:
            print(f"Bot {numeric_bot_id} is already registered in bot_rankings table.")
            print(f"Current values: rank_score={existing_rankings['rank_score']}, is_active={existing_rankings['is_active']}")
        else:
            # Insert initial rankings record with default values
            await conn.execute("""
                INSERT INTO bot_rankings (bot_id, rank_score, rank, timestamp, is_active)
                VALUES ($1, 50.0, 999, NOW(), true)
                """, numeric_bot_id)
            print(f"Successfully registered bot {numeric_bot_id} in bot_rankings with default rank score.")
        
        print(f"Bot {numeric_bot_id} is now fully registered in the system.")
        
    except Exception as e:
        print(f"Error registering bot: {e}")
    finally:
        if 'conn' in locals():
            await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a new bot in the database.")
    parser.add_argument("--db_url", required=True, help="Database connection URL (e.g., postgres://user:pass@localhost:5432/yourdb)")
    parser.add_argument("--bot_id", required=True, help="Bot ID (e.g., 17 or 17-126)")
    parser.add_argument("--ticker", help="Ticker symbol for the bot (e.g., TSLA). If not provided, will try to guess from bot_id.")
    parser.add_argument("--algo_id", required=True, help="Algorithm ID for the bot (e.g., 1)")
    args = parser.parse_args()
    
    # Convert algo_id to integer if it's a string
    if isinstance(args.algo_id, str) and args.algo_id.isdigit():
        args.algo_id = int(args.algo_id)
    
    # If ticker is not provided, try to guess it from the bot_id
    if not args.ticker:
        args.ticker = guess_ticker_from_bot_id(args.bot_id)
        print(f"No ticker provided. Using guessed ticker: {args.ticker}")

    asyncio.run(register_bot(args.db_url, args.bot_id, args.ticker, args.algo_id)) 