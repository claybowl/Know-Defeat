"""
WebSocket Trade Updates Test Script

This script sends test trade notifications through the WebSocket server 
to test the real-time updates in the rankings UI.

Usage:
    python tests/test_websocket_trade_updates.py
"""

import asyncio
import json
import random
import asyncpg
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data'
}

async def send_test_trade_opened(bot_id, ticker, price, trade_direction):
    """Send a test trade_opened notification through PostgreSQL."""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Create trade data
        trade_data = {
            'action': 'trade_opened',
            'trade_id': random.randint(10000, 99999),
            'bot_id': bot_id,
            'ticker': ticker,
            'entry_price': price,
            'trade_size': 1000.0,
            'trade_direction': trade_direction,
            'entry_time': datetime.now().isoformat(),
            'trade_status': 'open',
            'trailing_stop_price': price * (0.99 if trade_direction == 'long' else 1.01)
        }
        
        # Convert to JSON
        payload = json.dumps(trade_data)
        
        # Send notification via PostgreSQL
        await conn.execute(f"NOTIFY trade_channel, '{payload}'")
        
        logger.info(f"Sent test trade_opened notification for Bot {bot_id}")
        
        # Close connection
        await conn.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error sending trade notification: {e}")
        return False

async def send_test_trade_closed(bot_id, ticker, entry_price, exit_price, trade_direction):
    """Send a test trade_closed notification through PostgreSQL."""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Calculate P&L
        if trade_direction == 'long':
            pnl = (exit_price - entry_price) * 1000.0
            pnl_percent = ((exit_price / entry_price) - 1) * 100
        else:  # short
            pnl = (entry_price - exit_price) * 1000.0
            pnl_percent = ((entry_price / exit_price) - 1) * 100
        
        # Create trade data
        trade_data = {
            'action': 'trade_closed',
            'trade_id': random.randint(10000, 99999),
            'bot_id': bot_id,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_size': 1000.0,
            'trade_direction': trade_direction,
            'entry_time': (datetime.now().timestamp() - 300),  # 5 minutes ago
            'exit_time': datetime.now().isoformat(),
            'trade_status': 'closed',
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': 'test_close'
        }
        
        # Convert to JSON
        payload = json.dumps(trade_data)
        
        # Send notification via PostgreSQL
        await conn.execute(f"NOTIFY trade_channel, '{payload}'")
        
        logger.info(f"Sent test trade_closed notification for Bot {bot_id}")
        
        # Close connection
        await conn.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error sending trade notification: {e}")
        return False

async def send_bulk_update(count=5):
    """Send a bulk update with multiple random active trades."""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Get random bot IDs from the database
        bot_rows = await conn.fetch(
            "SELECT bot_id, ticker FROM sim_bots ORDER BY RANDOM() LIMIT $1",
            count
        )
        
        if not bot_rows:
            logger.error("No bots found in database")
            await conn.close()
            return False
        
        # Create trade data for each bot
        trades = []
        for bot in bot_rows:
            direction = random.choice(['long', 'short'])
            price = round(random.uniform(50, 500), 2)
            trades.append({
                'bot_id': bot['bot_id'],
                'ticker': bot['ticker'],
                'entry_price': price,
                'trade_size': 1000.0,
                'trade_direction': direction,
                'entry_time': datetime.now().isoformat(),
                'trade_status': 'open',
                'trailing_stop_price': price * (0.99 if direction == 'long' else 1.01)
            })
        
        # Create update data
        update_data = {
            'action': 'trade_update',
            'trades': trades,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert to JSON
        payload = json.dumps(update_data)
        
        # Send notification via PostgreSQL
        await conn.execute(f"NOTIFY trade_channel, '{payload}'")
        
        logger.info(f"Sent bulk update with {len(trades)} active trades")
        
        # Close connection
        await conn.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error sending bulk trade update: {e}")
        return False

async def get_random_bot():
    """Get a random bot ID from the database."""
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Get random bot ID
        result = await conn.fetchrow("SELECT bot_id, ticker FROM sim_bots ORDER BY RANDOM() LIMIT 1")
        
        # Close connection
        await conn.close()
        
        if result:
            return result
        else:
            logger.error("No bots found in database")
            return None
    
    except Exception as e:
        logger.error(f"Error getting random bot: {e}")
        return None

async def interactive_mode():
    """Run in interactive mode to send test trades."""
    
    logger.info("=== WebSocket Trade Updates Test Tool ===")
    logger.info("This tool sends test trade notifications through the WebSocket server")
    logger.info("Make sure the WebSocket server is running: python -m src.websocket_server")
    logger.info("")
    
    while True:
        logger.info("\nSelect an action:")
        logger.info("1. Send random trade_opened notification")
        logger.info("2. Send random trade_closed notification")
        logger.info("3. Send bulk update with multiple active trades")
        logger.info("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            # Get a random bot
            bot = await get_random_bot()
            if not bot:
                continue
                
            # Create random trade details
            direction = random.choice(['long', 'short'])
            price = round(random.uniform(50, 500), 2)
            
            # Send trade_opened notification
            await send_test_trade_opened(
                bot['bot_id'],
                bot['ticker'],
                price,
                direction
            )
            
        elif choice == '2':
            # Get a random bot
            bot = await get_random_bot()
            if not bot:
                continue
                
            # Create random trade details
            direction = random.choice(['long', 'short'])
            entry_price = round(random.uniform(50, 500), 2)
            
            # Calculate exit price with a random P&L
            pnl_factor = random.uniform(-0.05, 0.05)  # -5% to +5%
            if direction == 'long':
                exit_price = entry_price * (1 + pnl_factor)
            else:
                exit_price = entry_price * (1 - pnl_factor)
            exit_price = round(exit_price, 2)
            
            # Send trade_closed notification
            await send_test_trade_closed(
                bot['bot_id'],
                bot['ticker'],
                entry_price,
                exit_price,
                direction
            )
            
        elif choice == '3':
            # Ask for number of trades
            try:
                count = int(input("Enter number of trades to include (1-10): "))
                count = max(1, min(10, count))  # Limit between 1-10
            except ValueError:
                count = 5  # Default
                
            # Send bulk update
            await send_bulk_update(count)
            
        elif choice == '4':
            logger.info("Exiting...")
            break
            
        else:
            logger.info("Invalid choice, please try again")

async def auto_mode(duration=60, interval=5):
    """Run in automatic mode to simulate trading activity."""
    
    logger.info(f"Running in auto mode for {duration} seconds with {interval} second intervals")
    
    end_time = datetime.now().timestamp() + duration
    while datetime.now().timestamp() < end_time:
        action = random.choice(['open', 'close', 'bulk'])
        
        if action == 'open':
            # Get a random bot
            bot = await get_random_bot()
            if bot:
                # Create random trade details
                direction = random.choice(['long', 'short'])
                price = round(random.uniform(50, 500), 2)
                
                # Send trade_opened notification
                await send_test_trade_opened(
                    bot['bot_id'],
                    bot['ticker'],
                    price,
                    direction
                )
                
        elif action == 'close':
            # Get a random bot
            bot = await get_random_bot()
            if bot:
                # Create random trade details
                direction = random.choice(['long', 'short'])
                entry_price = round(random.uniform(50, 500), 2)
                
                # Calculate exit price with a random P&L
                pnl_factor = random.uniform(-0.05, 0.05)  # -5% to +5%
                if direction == 'long':
                    exit_price = entry_price * (1 + pnl_factor)
                else:
                    exit_price = entry_price * (1 - pnl_factor)
                exit_price = round(exit_price, 2)
                
                # Send trade_closed notification
                await send_test_trade_closed(
                    bot['bot_id'],
                    bot['ticker'],
                    entry_price,
                    exit_price,
                    direction
                )
                
        elif action == 'bulk':
            # Send bulk update with 3-8 trades
            count = random.randint(3, 8)
            await send_bulk_update(count)
        
        # Wait for next action
        await asyncio.sleep(interval)
    
    logger.info("Auto mode complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Trade Updates Test Tool")
    parser.add_argument(
        "--auto", 
        action="store_true", 
        help="Run in automatic mode (non-interactive)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=60,
        help="Duration in seconds for auto mode (default: 60)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5,
        help="Interval in seconds between actions in auto mode (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        asyncio.run(auto_mode(args.duration, args.interval))
    else:
        asyncio.run(interactive_mode()) 