"""
Script to monitor and retry completion of pending_exit trades.
Run this script alongside your bots to automatically close trades that get stuck in pending_exit state.
"""

import asyncio
import logging
import asyncpg
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PendingTradeMonitor")

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def close_pending_trade(conn, trade):
    """Close a single pending trade."""
    try:
        # Convert values to float to avoid type mismatches
        exit_price = float(trade['exit_trigger_price'])
        entry_price = float(trade['entry_price'])
        trade_size = float(trade['trade_size'])
        
        # Calculate PnL
        if trade['trade_direction'] == 'LONG':
            pnl = (exit_price - entry_price) * (trade_size / entry_price)
        else:  # SHORT
            pnl = (entry_price - exit_price) * (trade_size / entry_price)
        
        # Update the trade status to closed
        await conn.execute("""
            UPDATE sim_bot_trades
            SET 
                trade_status = 'closed',
                exit_time = exit_trigger_time,
                exit_price = exit_trigger_price,
                trade_pnl = $1
            WHERE trade_id = $2
        """, pnl, trade['trade_id'])
        
        logger.info(f"Closed trade {trade['trade_id']} for Bot {trade['bot_id']} with PnL: ${pnl:.2f}")
        return True
    except Exception as e:
        logger.error(f"Error closing trade {trade['trade_id']}: {e}")
        return False

async def monitor_pending_trades():
    """Continuously monitor for trades stuck in pending_exit status and close them."""
    try:
        # Connect to the database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        logger.info("Starting pending trade monitor")
        
        while True:
            try:
                # Get all trades stuck in pending_exit for more than 5 seconds
                pending_trades = await conn.fetch("""
                    SELECT 
                        trade_id, bot_id, ticker, entry_price, exit_trigger_price, 
                        trade_direction, trade_size, exit_trigger_time
                    FROM sim_bot_trades
                    WHERE trade_status = 'pending_exit'
                    AND exit_trigger_time < NOW() - INTERVAL '5 seconds'
                """)
                
                if pending_trades:
                    logger.info(f"Found {len(pending_trades)} trades stuck in pending_exit status")
                    
                    for trade in pending_trades:
                        await close_pending_trade(conn, trade)
                
                # Wait before next check
                await asyncio.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(30)  # Longer wait on error
    
    except Exception as e:
        logger.error(f"Main monitor error: {e}")
    finally:
        # Close the database connection
        await conn.close()
        logger.info("Pending trade monitor stopped")

if __name__ == "__main__":
    try:
        asyncio.run(monitor_pending_trades())
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")