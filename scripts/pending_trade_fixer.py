"""
Pending Trade Fixer - Utility to fix trades stuck in pending_exit status

Run this script to find and fix trades that are stuck in pending_exit status.
The script will properly close these trades and calculate their P&L.
"""

import asyncio
import asyncpg
import sys
import os
import time
import logging
from datetime import datetime, timedelta
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pending_trade_fixer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PendingTradeFixer")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

async def get_pending_trades(conn, min_age_seconds=10):
    """Get trades stuck in pending_exit status"""
    return await conn.fetch("""
        SELECT trade_id, bot_id, ticker, entry_price, exit_trigger_price, 
               trade_direction, trade_size, exit_trigger_time
        FROM sim_bot_trades
        WHERE trade_status = 'pending_exit'
        AND exit_trigger_time < NOW() - INTERVAL '{}' SECOND
        ORDER BY exit_trigger_time ASC
    """.format(min_age_seconds))

async def get_bot_name(conn, bot_id):
    """Get bot name from database"""
    name = await conn.fetchval(
        "SELECT name FROM sim_bots WHERE bot_id = $1", 
        bot_id
    )
    return name or f"Bot-{bot_id}"

async def close_pending_trade(conn, trade, fix_time=None):
    """Close a single pending trade"""
    try:
        # Convert values to float to avoid type mismatches
        exit_price = float(trade['exit_trigger_price'])
        entry_price = float(trade['entry_price'])
        trade_size = float(trade['trade_size'])
        
        # Calculate P&L
        if trade['trade_direction'] == 'LONG':
            pnl = (exit_price - entry_price) * (trade_size / entry_price)
            pnl_percent = ((exit_price / entry_price) - 1) * 100
        else:  # SHORT
            pnl = (entry_price - exit_price) * (trade_size / entry_price)
            pnl_percent = ((entry_price / exit_price) - 1) * 100
        
        # Use specified fix time or exit_trigger_time
        exit_time = fix_time or trade['exit_trigger_time']
        
        # Check if pnl_percent column exists
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'sim_bot_trades' AND column_name = 'pnl_percent'
            )
        """)
        
        # Update the trade status to closed
        if column_exists:
            await conn.execute("""
                UPDATE sim_bot_trades
                SET 
                    trade_status = 'closed',
                    exit_time = $1,
                    exit_price = exit_trigger_price,
                    trade_pnl = $2,
                    pnl_percent = $3
                WHERE trade_id = $4
            """, exit_time, pnl, pnl_percent, trade['trade_id'])
        else:
            # If pnl_percent column doesn't exist, don't include it in the update
            await conn.execute("""
                UPDATE sim_bot_trades
                SET 
                    trade_status = 'closed',
                    exit_time = $1,
                    exit_price = exit_trigger_price,
                    trade_pnl = $2
                WHERE trade_id = $3
            """, exit_time, pnl, trade['trade_id'])
        
        bot_name = await get_bot_name(conn, trade['bot_id'])
        
        logger.info(f"{Colors.GREEN}Fixed trade {trade['trade_id']} for {bot_name} ({trade['ticker']}) with P&L: ${pnl:.2f} ({pnl_percent:.2f}%){Colors.RESET}")
        return True
    except Exception as e:
        logger.error(f"{Colors.RED}Error closing trade {trade['trade_id']}: {e}{Colors.RESET}")
        return False

async def fix_pending_trades(min_age_seconds=10, auto_fix=False, specific_trade_id=None):
    """Find and fix trades stuck in pending_exit status"""
    try:
        # Connect to the database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info(f"{Colors.BOLD}Connected to database{Colors.RESET}")
        
        # Get pending trades based on criteria
        if specific_trade_id:
            # Get a specific trade
            pending_trades = await conn.fetch("""
                SELECT trade_id, bot_id, ticker, entry_price, exit_trigger_price, 
                       trade_direction, trade_size, exit_trigger_time
                FROM sim_bot_trades
                WHERE trade_id = $1 AND trade_status = 'pending_exit'
            """, specific_trade_id)
            
            if not pending_trades:
                logger.error(f"{Colors.RED}Trade {specific_trade_id} not found or not in pending_exit status{Colors.RESET}")
                await conn.close()
                return
        else:
            # Get all pending trades based on age
            pending_trades = await get_pending_trades(conn, min_age_seconds)
        
        if not pending_trades:
            logger.info(f"{Colors.GREEN}No trades found in pending_exit status{Colors.RESET}")
            await conn.close()
            return
        
        # Display pending trades
        logger.info(f"{Colors.YELLOW}Found {len(pending_trades)} trades in pending_exit status{Colors.RESET}")
        
        for i, trade in enumerate(pending_trades, 1):
            bot_name = await get_bot_name(conn, trade['bot_id'])
            stuck_duration = datetime.now() - trade['exit_trigger_time']
            
            logger.info(f"{Colors.CYAN}[{i}/{len(pending_trades)}] Trade {trade['trade_id']}:{Colors.RESET}")
            logger.info(f"  Bot: {trade['bot_id']} ({bot_name})")
            logger.info(f"  Ticker: {trade['ticker']}")
            logger.info(f"  Direction: {trade['trade_direction']}")
            logger.info(f"  Entry Price: ${float(trade['entry_price']):.2f}")
            logger.info(f"  Exit Trigger Price: ${float(trade['exit_trigger_price']):.2f}")
            logger.info(f"  Stuck for: {stuck_duration}")
        
        # Fix trades based on mode
        if auto_fix:
            logger.info(f"{Colors.BOLD}Auto-fixing all {len(pending_trades)} pending trades...{Colors.RESET}")
            
            fixed_count = 0
            for trade in pending_trades:
                if await close_pending_trade(conn, trade):
                    fixed_count += 1
            
            logger.info(f"{Colors.GREEN}Successfully fixed {fixed_count}/{len(pending_trades)} trades{Colors.RESET}")
        
        elif specific_trade_id:
            # Fix specific trade
            trade = pending_trades[0]  # There should be only one trade in the list
            if await close_pending_trade(conn, trade):
                logger.info(f"{Colors.GREEN}Successfully fixed trade {specific_trade_id}{Colors.RESET}")
        
        else:
            # Interactive mode - ask for confirmation
            logger.info(f"{Colors.YELLOW}Do you want to fix these trades? (y/n){Colors.RESET}")
            response = input().strip().lower()
            
            if response == 'y':
                fixed_count = 0
                for trade in pending_trades:
                    if await close_pending_trade(conn, trade):
                        fixed_count += 1
                
                logger.info(f"{Colors.GREEN}Successfully fixed {fixed_count}/{len(pending_trades)} trades{Colors.RESET}")
            else:
                logger.info("No trades were fixed")
        
        # Close the database connection
        await conn.close()
        
    except Exception as e:
        logger.error(f"{Colors.RED}Error fixing pending trades: {e}{Colors.RESET}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fix trades stuck in pending_exit status")
    
    parser.add_argument("--min-age", type=int, default=10,
                        help="Minimum age in seconds for a trade to be considered stuck (default: 10)")
    
    parser.add_argument("--auto", action="store_true",
                        help="Automatically fix all stuck trades without asking for confirmation")
    
    parser.add_argument("--trade-id", type=int,
                        help="Fix a specific trade by ID")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Print banner
    print(f"""
{Colors.BOLD}{Colors.BLUE}PENDING TRADE FIXER{Colors.RESET}

This utility will find and fix trades stuck in pending_exit status.
""")
    
    # Run the fixer
    asyncio.run(fix_pending_trades(
        min_age_seconds=args.min_age,
        auto_fix=args.auto,
        specific_trade_id=args.trade_id
    ))