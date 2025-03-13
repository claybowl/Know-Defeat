#!/usr/bin/env python
"""
Bot Registration Script

This script reads bot YAML definitions from the src/bots directory
and registers them in the database.
"""

import asyncio
import sys
import os
import logging

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ib_controller_simple import DataIngestionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('register_bots.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def register_bots(bot_dir="src/bots"):
    """Register all bots from the specified directory."""
    logger.info(f"Starting bot registration from directory: {bot_dir}")
    
    # Create a manager instance with empty symbol list (we only need DB functionality)
    manager = DataIngestionManager([])
    
    try:
        # Initialize the database
        await manager.init_db()
        
        # Register bots
        result = await manager.register_bots_from_directory(bot_dir)
        
        if result["success"]:
            logger.info(f"Successfully registered {result['registered']} bots")
            logger.info(f"Skipped {result['skipped']} bots")
        else:
            logger.error(f"Failed to register bots: {result.get('error', 'Unknown error')}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error registering bots: {e}")
        return False
    finally:
        # Close database connection
        if hasattr(manager, 'db_pool') and manager.db_pool:
            await manager.db_pool.close()
            logger.info("Database connection closed")

async def main():
    """Main function."""
    # Allow specifying a different bot directory
    bot_dir = "src/bots"
    if len(sys.argv) > 1:
        bot_dir = sys.argv[1]
    
    success = await register_bots(bot_dir)
    
    if success:
        logger.info("Bot registration completed successfully")
        sys.exit(0)
    else:
        logger.error("Bot registration failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 