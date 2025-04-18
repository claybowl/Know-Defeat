#!/usr/bin/env python
"""
This script applies the PostgreSQL notification trigger for bot_metrics changes.
Run this script to set up real-time notifications for bot metrics updates.
"""

import asyncio
import asyncpg
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection settings
DB_USER = "clayb"
DB_PASSWORD = "musicman"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "tick_data"

async def apply_notification_trigger():
    """Apply the notification trigger to the bot_metrics table."""
    # Get the path to the SQL file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sql_file_path = os.path.join(os.path.dirname(script_dir), "sql", "create_notification_trigger.sql")
    
    # Check if the SQL file exists
    if not os.path.exists(sql_file_path):
        logger.error(f"SQL file not found: {sql_file_path}")
        return False
    
    # Read the SQL file
    with open(sql_file_path, 'r') as f:
        sql = f.read()
    
    # Connect to the database
    try:
        conn = await asyncpg.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        
        # Execute the SQL script
        logger.info("Applying notification trigger to bot_metrics table...")
        await conn.execute(sql)
        logger.info("Notification trigger applied successfully.")
        
        # Test the trigger
        logger.info("Testing notification trigger...")
        
        # Set up a listener for the notification channel
        await conn.add_listener("bot_metrics_channel", lambda conn, pid, channel, payload: 
            logger.info(f"Received notification: {payload}"))
        
        # Update a bot metric to trigger the notification
        # This is just a dummy update to test the trigger
        await conn.execute("""
            WITH latest_bot AS (
                SELECT bot_id FROM bot_metrics LIMIT 1
            )
            UPDATE bot_metrics 
            SET last_updated = NOW() 
            WHERE bot_id = (SELECT bot_id FROM latest_bot)
        """)
        
        # Give some time for the notification to be processed
        await asyncio.sleep(1)
        
        # Clean up
        await conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Error applying notification trigger: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(apply_notification_trigger())
    if result:
        logger.info("Script completed successfully.")
    else:
        logger.error("Script failed.")
        exit(1) 