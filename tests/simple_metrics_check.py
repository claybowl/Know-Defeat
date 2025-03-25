"""
A simplified metrics system check script.
"""

import asyncio
import asyncpg
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

async def check_metrics_system():
    """Check the metrics system."""
    # Create a connection pool
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        # Get all bots
        async with pool.acquire() as conn:
            bots = await conn.fetch("SELECT bot_id, ticker, algorithm_type FROM sim_bots")
            
            if not bots:
                logger.error("No bots found in the database.")
                return
            
            logger.info(f"Found {len(bots)} bots in the database")
            
            # Check if each bot has trades
            print("\n=== Bot Trade Summary ===")
            print(f"{'Bot ID':^6} | {'Ticker':^6} | {'Algorithm':^15} | {'Total Trades':^12} | {'Open Trades':^10} | {'Has Metrics':^10}")
            print("-" * 75)
            
            for bot in bots:
                bot_id = bot['bot_id']
                ticker = bot['ticker']
                algorithm = bot['algorithm_type']
                
                # Count trades
                trade_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = $1",
                    bot_id
                )
                
                # Count open trades
                open_trades = await conn.fetchval(
                    "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = $1 AND trade_status = 'open'",
                    bot_id
                )
                
                # Check if metrics exist
                metrics_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM bot_metrics WHERE bot_id = $1",
                    bot_id
                )
                
                has_metrics = "Yes" if metrics_count > 0 else "No"
                
                print(f"{bot_id:^6} | {ticker:^6} | {algorithm:^15} | {trade_count:^12} | {open_trades:^10} | {has_metrics:^10}")
            
            # Show latest trades
            print("\n=== Latest Trades ===")
            latest_trades = await conn.fetch("""
                SELECT trade_id, bot_id, ticker, trade_direction, 
                       entry_price, exit_price, trade_pnl, entry_time, exit_time
                FROM sim_bot_trades
                ORDER BY entry_time DESC
                LIMIT 5
            """)
            
            if latest_trades:
                print(f"{'Trade ID':^8} | {'Bot ID':^6} | {'Ticker':^6} | {'Direction':^9} | {'Entry Price':^11} | {'Exit Price':^10} | {'PnL':^8} | {'Entry Time':^19} | {'Exit Time':^19}")
                print("-" * 120)
                
                for trade in latest_trades:
                    entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if trade['entry_time'] else 'N/A'
                    exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if trade['exit_time'] else 'N/A'
                    
                    entry_price = f"{float(trade['entry_price']):.2f}" if trade['entry_price'] else 'N/A'
                    exit_price = f"{float(trade['exit_price']):.2f}" if trade['exit_price'] else 'N/A'
                    pnl = f"{float(trade['trade_pnl']):.2f}" if trade['trade_pnl'] else 'N/A'
                    
                    print(f"{trade['trade_id']:^8} | {trade['bot_id']:^6} | {trade['ticker']:^6} | "
                          f"{trade['trade_direction']:^9} | {entry_price:^11} | "
                          f"{exit_price:^10} | {pnl:^8} | "
                          f"{entry_time:^19} | {exit_time:^19}")
            else:
                print("No trades found in the database.")
            
            # Show latest metrics
            print("\n=== Latest Metrics ===")
            latest_metrics = await conn.fetch("""
                SELECT DISTINCT ON (bot_id) 
                    bot_id, ticker, total_trades, avg_win_rate, total_pnl, timestamp
                FROM bot_metrics
                ORDER BY bot_id, timestamp DESC
                LIMIT 10
            """)
            
            if latest_metrics:
                print(f"{'Bot ID':^6} | {'Ticker':^6} | {'Total Trades':^12} | {'Win Rate':^10} | {'Total PnL':^10} | {'Updated':^19}")
                print("-" * 80)
                
                for metric in latest_metrics:
                    updated = metric['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if metric['timestamp'] else 'N/A'
                    win_rate = f"{float(metric['avg_win_rate']):.2f}%" if metric['avg_win_rate'] else 'N/A'
                    total_pnl = f"${float(metric['total_pnl']):.2f}" if metric['total_pnl'] else 'N/A'
                    
                    print(f"{metric['bot_id']:^6} | {metric['ticker']:^6} | {metric['total_trades']:^12} | "
                          f"{win_rate:^10} | {total_pnl:^10} | {updated:^19}")
            else:
                print("No metrics found in the database.")
    
    except Exception as e:
        logger.error(f"Error in metrics system check: {e}")
    
    finally:
        # Close the connection pool
        await pool.close()

if __name__ == "__main__":
    asyncio.run(check_metrics_system())