"""
Close Bot Trades Script

This script will close all open trades for a specific bot.
"""

import asyncio
import logging
import asyncpg
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_emergency.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("close_bot_trades")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def get_db_pool():
    """Create and return a database connection pool."""
    return await asyncpg.create_pool(**DB_CONFIG)

async def get_latest_price(ticker):
    """Get the latest valid price for a ticker."""
    db_pool = await get_db_pool()
    try:
        async with db_pool.acquire() as conn:
            price_row = await conn.fetchrow("""
                SELECT price 
                FROM tick_data 
                WHERE ticker = $1 AND price > 0
                ORDER BY timestamp DESC 
                LIMIT 1
            """, ticker)
            
            if price_row:
                price = price_row['price']
                logger.info(f"Latest price for {ticker}: ${price:.2f}")
                return price
            else:
                logger.warning(f"No valid price found for {ticker}")
                return None
    finally:
        await db_pool.close()

async def close_bot_trades(bot_id):
    """Close all open trades for a specific bot."""
    db_pool = await get_db_pool()
    try:
        async with db_pool.acquire() as conn:
            # Get all open trades for this bot
            open_trades = await conn.fetch("""
                SELECT trade_id, ticker, entry_price, trade_direction, trade_size
                FROM sim_bot_trades
                WHERE bot_id = $1 AND trade_status = 'open'
            """, bot_id)
            
            logger.info(f"Found {len(open_trades)} open trades for bot {bot_id}")
            
            # Close each trade
            for trade in open_trades:
                trade_id = trade['trade_id']
                ticker = trade['ticker']
                entry_price = trade['entry_price']
                
                # Get latest price
                exit_price = await get_latest_price(ticker)
                if exit_price is None:
                    exit_price = entry_price  # Use entry price if no valid price found
                
                # Calculate PnL
                if trade['trade_direction'] == 'LONG':
                    pnl = (exit_price - entry_price) * (trade['trade_size'] / entry_price)
                else:  # SHORT
                    pnl = (entry_price - exit_price) * (trade['trade_size'] / entry_price)
                
                # Update the trade to closed status
                await conn.execute("""
                    UPDATE sim_bot_trades
                    SET 
                        trade_status = 'closed',
                        exit_time = NOW(),
                        exit_price = $1,
                        trade_pnl = $2
                    WHERE trade_id = $3
                """, exit_price, pnl, trade_id)
                
                logger.info(f"Closed trade {trade_id} for Bot {bot_id} ({ticker}) with PnL: ${pnl:.2f}")
            
            return len(open_trades)
    finally:
        await db_pool.close()

async def display_bot_trades(bot_id):
    """Display all open trades for a specific bot."""
    db_pool = await get_db_pool()
    try:
        async with db_pool.acquire() as conn:
            # Get all open trades for this bot
            open_trades = await conn.fetch("""
                SELECT trade_id, ticker, entry_price, trade_direction, trade_size, entry_time
                FROM sim_bot_trades
                WHERE bot_id = $1 AND trade_status = 'open'
                ORDER BY entry_time DESC
            """, bot_id)
            
            if not open_trades:
                logger.info(f"No open trades found for bot {bot_id}")
                return False
                
            logger.info(f"Found {len(open_trades)} open trades for bot {bot_id}:")
            logger.info("=" * 80)
            logger.info(f"{'ID':<5} {'Ticker':<6} {'Direction':<8} {'Entry Price':<12} {'Size':<10} {'Entry Time':<25}")
            logger.info("-" * 80)
            
            for trade in open_trades:
                logger.info(
                    f"{trade['trade_id']:<5} "
                    f"{trade['ticker']:<6} "
                    f"{trade['trade_direction']:<8} "
                    f"${trade['entry_price']:<11.2f} "
                    f"${trade['trade_size']:<9.2f} "
                    f"{trade['entry_time']!s:<25}"
                )
            
            logger.info("=" * 80)
            return True
    finally:
        await db_pool.close()

async def main():
    """Main function to close trades for a specific bot."""
    parser = argparse.ArgumentParser(description='Close all open trades for a specific bot')
    parser.add_argument('bot_id', type=int, help='The ID of the bot whose trades should be closed')
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting trade closure for bot {args.bot_id}")
        
        # First display all open trades for this bot
        has_trades = await display_bot_trades(args.bot_id)
        
        if not has_trades:
            logger.info(f"No trades to close for bot {args.bot_id}")
            return
            
        # Ask for confirmation
        print(f"\nAre you sure you want to close all trades for bot {args.bot_id}? (yes/no): ", end="")
        response = input().strip().lower()
        
        if response != "yes":
            logger.info("Trade closure cancelled by user")
            return
            
        # Close the trades
        closed_count = await close_bot_trades(args.bot_id)
        logger.info(f"Successfully closed {closed_count} trades for bot {args.bot_id}")
    except Exception as e:
        logger.error(f"Error closing trades: {e}")
    
if __name__ == "__main__":
    asyncio.run(main()) 