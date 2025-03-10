"""
Close All Trades Script

This script will close all open trades in the database.
Use this for emergency situations when you need to exit all positions.
"""

import asyncio
import logging
import asyncpg
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

logger = logging.getLogger("close_all_trades")

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

async def get_latest_prices():
    """Get the latest valid prices for all tickers."""
    db_pool = await get_db_pool()
    try:
        async with db_pool.acquire() as conn:
            # For each ticker, get the most recent price that is > 0
            tickers = await conn.fetch("""
                SELECT DISTINCT ticker FROM tick_data
            """)
            
            latest_prices = {}
            for ticker_row in tickers:
                ticker = ticker_row['ticker']
                price_row = await conn.fetchrow("""
                    SELECT price 
                    FROM tick_data 
                    WHERE ticker = $1 AND price > 0
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, ticker)
                
                if price_row:
                    latest_prices[ticker] = price_row['price']
                    logger.info(f"Latest price for {ticker}: ${latest_prices[ticker]:.2f}")
                else:
                    logger.warning(f"No valid price found for {ticker}")
            
            return latest_prices
    finally:
        await db_pool.close()

async def display_open_trades():
    """Display all currently open trades."""
    db_pool = await get_db_pool()
    try:
        async with db_pool.acquire() as conn:
            # Get all open trades
            open_trades = await conn.fetch("""
                SELECT t.trade_id, t.bot_id, t.ticker, t.entry_price, t.trade_direction, 
                       t.trade_size, t.entry_time, b.rank_score
                FROM sim_bot_trades t
                LEFT JOIN bot_rankings b ON t.bot_id = b.bot_id
                WHERE t.trade_status = 'open'
                ORDER BY b.rank_score DESC
            """)
            
            if not open_trades:
                logger.info("No open trades found")
                return
                
            logger.info(f"Found {len(open_trades)} open trades:")
            logger.info("=" * 80)
            logger.info(f"{'ID':<5} {'Bot':<5} {'Ticker':<6} {'Direction':<8} {'Entry Price':<12} {'Size':<10} {'Entry Time':<25} {'Bot Rank':<10}")
            logger.info("-" * 80)
            
            for trade in open_trades:
                logger.info(
                    f"{trade['trade_id']:<5} "
                    f"{trade['bot_id']:<5} "
                    f"{trade['ticker']:<6} "
                    f"{trade['trade_direction']:<8} "
                    f"${trade['entry_price']:<11.2f} "
                    f"${trade['trade_size']:<9.2f} "
                    f"{trade['entry_time']!s:<25} "
                    f"{trade['rank_score'] if trade['rank_score'] else 'N/A':<10}"
                )
            
            logger.info("=" * 80)
    finally:
        await db_pool.close()

async def close_all_trades():
    """Close all open trades in the database."""
    db_pool = await get_db_pool()
    try:
        # Get latest valid prices for all tickers
        latest_prices = await get_latest_prices()
        
        async with db_pool.acquire() as conn:
            # Get all open trades
            open_trades = await conn.fetch("""
                SELECT trade_id, bot_id, ticker, entry_price, trade_direction, trade_size
                FROM sim_bot_trades
                WHERE trade_status = 'open'
            """)
            
            logger.info(f"Found {len(open_trades)} open trades to close")
            
            # Close each trade
            for trade in open_trades:
                trade_id = trade['trade_id']
                ticker = trade['ticker']
                entry_price = trade['entry_price']
                
                # Use latest price if available, otherwise use entry price
                exit_price = latest_prices.get(ticker, entry_price)
                
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
                
                logger.info(f"Closed trade {trade_id} for Bot {trade['bot_id']} ({ticker}) with PnL: ${pnl:.2f}")
            
            # Update bot activations to reflect the new state
            await conn.execute("""
                UPDATE bot_rankings
                SET is_active = true
            """)
            
            logger.info("All bots set to active state")
            
            return len(open_trades)
    finally:
        await db_pool.close()

async def main():
    """Main function to close all trades."""
    try:
        logger.info("Starting emergency trade closure")
        
        # First display all open trades
        await display_open_trades()
        
        # Ask for confirmation
        print("\nAre you sure you want to close ALL open trades? (yes/no): ", end="")
        response = input().strip().lower()
        
        if response != "yes":
            logger.info("Trade closure cancelled by user")
            return
            
        # Close all trades
        closed_count = await close_all_trades()
        logger.info(f"Successfully closed {closed_count} trades")
    except Exception as e:
        logger.error(f"Error closing trades: {e}")
    
if __name__ == "__main__":
    asyncio.run(main()) 