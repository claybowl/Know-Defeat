"""
Script to close all pending_exit trades in the database.
This will update all trades with status 'pending_exit' to 'closed' and calculate PnL.
"""

import asyncio
import logging
import asyncpg
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def close_pending_trades():
    """Close all trades with 'pending_exit' status in the database."""
    try:
        # Connect to the database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Get all pending_exit trades
        pending_trades = await conn.fetch("""
            SELECT trade_id, bot_id, ticker, entry_price, exit_trigger_price, 
                   trade_direction, trade_size
            FROM sim_bot_trades
            WHERE trade_status = 'pending_exit'
        """)
        
        logger.info(f"Found {len(pending_trades)} trades in pending_exit status")
        
        for trade in pending_trades:
            # Calculate PnL
            if trade['trade_direction'] == 'LONG':
                pnl = (trade['exit_trigger_price'] - trade['entry_price']) * (trade['trade_size'] / trade['entry_price'])
            else:  # SHORT
                pnl = (trade['entry_price'] - trade['exit_trigger_price']) * (trade['trade_size'] / trade['entry_price'])
            
            # Update the trade status to closed
            await conn.execute("""
                UPDATE sim_bot_trades
                SET 
                    trade_status = 'closed',
                    exit_price = exit_trigger_price,
                    trade_pnl = $1
                WHERE trade_id = $2
            """, pnl, trade['trade_id'])
            
            logger.info(f"Closed trade {trade['trade_id']} for Bot {trade['bot_id']} with PnL: ${pnl:.2f}")
        
        # Close the database connection
        await conn.close()
        
        logger.info(f"Successfully closed {len(pending_trades)} pending trades")
        
    except Exception as e:
        logger.error(f"Error closing pending trades: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(close_pending_trades())