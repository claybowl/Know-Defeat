#!/usr/bin/env python
"""
Test script for the bot_metrics WebSocket implementation.
This script:
1. Connects to the WebSocket endpoint
2. Listens for messages
3. Updates a bot metric in the database
4. Verifies that an update notification is received
"""

import asyncio
import websockets
import json
import logging
import asyncpg
import sys
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WS_URL = "ws://localhost:8000/api/ws/metrics"
DB_USER = "clayb"
DB_PASSWORD = "musicman"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "tick_data"

# Globals
received_messages = []
test_bot_id = None

async def listen_for_websocket_messages():
    """Connect to the WebSocket and listen for messages."""
    try:
        async with websockets.connect(WS_URL) as websocket:
            logger.info(f"Connected to WebSocket at {WS_URL}")
            
            # Listen for initial message (should be current metrics)
            initial_msg = await websocket.recv()
            initial_data = json.loads(initial_msg)
            logger.info(f"Received initial data with {len(initial_data)} bot metrics")
            received_messages.append(initial_data)
            
            # Keep listening for 10 seconds
            end_time = time.time() + 10
            while time.time() < end_time:
                try:
                    # Set a timeout for the recv to not block indefinitely
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(msg)
                    logger.info(f"Received update notification for bot_id: {data[0]['bot_id']}")
                    received_messages.append(data)
                except asyncio.TimeoutError:
                    # This is expected when no messages arrive
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error receiving WebSocket message: {e}")
                    break
                    
            logger.info(f"Finished listening. Received {len(received_messages)} total messages.")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

async def update_bot_metric():
    """Update a bot metric in the database to trigger a notification."""
    try:
        # Wait a bit to ensure WebSocket connection is established
        await asyncio.sleep(2)
        
        conn = await asyncpg.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        
        # Get a bot_id to update
        bot_record = await conn.fetchrow("SELECT bot_id FROM bot_metrics LIMIT 1")
        if not bot_record:
            logger.error("No bot metrics found in the database")
            await conn.close()
            return False
            
        global test_bot_id
        test_bot_id = bot_record['bot_id']
        
        # Update the bot metric
        logger.info(f"Updating bot_id {test_bot_id} to trigger notification...")
        await conn.execute("""
            UPDATE bot_metrics 
            SET last_updated = $1
            WHERE bot_id = $2
        """, datetime.now(), test_bot_id)
        
        # Wait a bit for the notification to process
        await asyncio.sleep(1)
        
        # Update another field to trigger another notification
        logger.info(f"Updating total_pnl for bot_id {test_bot_id}...")
        await conn.execute("""
            UPDATE bot_metrics 
            SET total_pnl = total_pnl + 0.01
            WHERE bot_id = $1
        """, test_bot_id)
        
        await conn.close()
        return True
    except Exception as e:
        logger.error(f"Error updating bot metric: {e}")
        return False

async def run_test():
    """Run the WebSocket test."""
    # Start the WebSocket listener
    ws_task = asyncio.create_task(listen_for_websocket_messages())
    
    # Wait a moment for connection to establish
    await asyncio.sleep(2)
    
    # Update a bot metric
    success = await update_bot_metric()
    if not success:
        logger.error("Failed to update bot metric")
        return False
    
    # Wait for the WebSocket listener to complete
    await ws_task
    
    # Verify results
    update_received = False
    for msgs in received_messages[1:]:  # Skip the initial message
        for msg in msgs:
            if msg.get('bot_id') == test_bot_id:
                update_received = True
                logger.info(f"✅ Received expected update for bot_id {test_bot_id}")
                break
    
    if not update_received:
        logger.error(f"❌ Did not receive expected update for bot_id {test_bot_id}")
        return False
        
    return True

if __name__ == "__main__":
    logger.info("Starting WebSocket notification test...")
    result = asyncio.run(run_test())
    if result:
        logger.info("✅ Test completed successfully: WebSocket notifications are working!")
        sys.exit(0)
    else:
        logger.error("❌ Test failed: WebSocket notifications are not working properly")
        sys.exit(1) 