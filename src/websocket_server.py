import asyncio
import json
import logging
import argparse
import signal
from typing import Dict, Set, Any

import asyncpg
from websockets.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from src.db_connection import create_db_pool

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# WebSocket connections by channel
connections: Dict[str, Set] = {
    'bot_metrics': set(),
    'trades': set(),
    'rankings': set()
}

# Channel mappings from PostgreSQL to client channels
channel_mapping = {
    'bot_metrics_channel': 'bot_metrics',
    'trade_channel': 'trades',
    'ranking_channel': 'rankings'
}

async def db_listener(ws_server):
    """Listen for PostgreSQL notifications and forward to WebSocket clients."""
    pool = await create_db_pool()
    try:
        # Get a connection for listening
        conn = await pool.acquire()
        
        # Set up notification listeners
        await conn.add_listener('bot_metrics_channel', on_pg_notification)
        await conn.add_listener('trade_channel', on_pg_notification)
        await conn.add_listener('ranking_channel', on_pg_notification)
        
        logger.info("PostgreSQL notification listener started")
        
        # Keep running until signaled to stop
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Database listener cancelled")
    except Exception as e:
        logger.error(f"Error in database listener: {e}")
    finally:
        # Clean up listeners
        for channel in ['bot_metrics_channel', 'trade_channel', 'ranking_channel']:
            await conn.remove_listener(channel, on_pg_notification)
        
        await pool.release(conn)
        await pool.close()
        logger.info("Database listener stopped")

async def on_pg_notification(connection, pid, channel, payload):
    """Handle PostgreSQL notifications and forward to WebSocket clients."""
    try:
        # Parse JSON payload
        data = json.loads(payload)
        
        # Map PostgreSQL channel to WebSocket channel
        ws_channel = channel_mapping.get(channel)
        if not ws_channel:
            logger.warning(f"Unknown channel mapping for: {channel}")
            return
            
        # Get all WebSocket connections for this channel
        clients = connections.get(ws_channel, set())
        if not clients:
            return
            
        # Prepare message for clients
        message = json.dumps({
            'channel': ws_channel,
            'data': data,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # Broadcast to all clients
        disconnected = set()
        for websocket in clients:
            try:
                await websocket.send(message)
            except ConnectionClosedOK:
                disconnected.add(websocket)
            except ConnectionClosedError:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(websocket)
                
        # Remove disconnected clients
        for websocket in disconnected:
            clients.remove(websocket)
            
        logger.debug(f"Notification broadcast to {len(clients)} clients on channel {ws_channel}")
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in notification payload: {payload}")
    except Exception as e:
        logger.error(f"Error processing notification: {e}")

async def websocket_handler(websocket, path):
    """Handle WebSocket connections and subscribe to channels."""
    # Default to bot_metrics channel if path not specified
    channel = path.strip('/') or 'bot_metrics'
    
    # Register connection
    if channel not in connections:
        connections[channel] = set()
    connections[channel].add(websocket)
    
    logger.info(f"New WebSocket connection to channel: {channel}")
    
    try:
        # Initial message to client
        await websocket.send(json.dumps({
            'type': 'connection_established',
            'channel': channel,
            'message': f'Connected to {channel} channel'
        }))
        
        # Keep connection alive until client disconnects
        async for message in websocket:
            # Client can send messages to control subscription
            try:
                data = json.loads(message)
                if 'subscribe' in data:
                    new_channel = data['subscribe']
                    # Remove from current channel
                    if websocket in connections[channel]:
                        connections[channel].remove(websocket)
                    
                    # Add to new channel
                    if new_channel not in connections:
                        connections[new_channel] = set()
                    connections[new_channel].add(websocket)
                    channel = new_channel
                    
                    await websocket.send(json.dumps({
                        'type': 'subscription_changed',
                        'channel': channel,
                        'message': f'Now subscribed to {channel} channel'
                    }))
            except json.JSONDecodeError:
                logger.warning(f"Invalid message from client: {message}")
    except ConnectionClosedOK:
        logger.info("WebSocket connection closed normally")
    except ConnectionClosedError:
        logger.info("WebSocket connection closed with error")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        # Unregister connection
        if channel in connections and websocket in connections[channel]:
            connections[channel].remove(websocket)
        logger.info(f"WebSocket connection closed: {channel}")

async def start_server(host='0.0.0.0', port=8765):
    """Start the WebSocket server and database listener."""
    websocket_server = await websocket_serve(
        websocket_handler,
        host,
        port,
        ping_interval=30,
        ping_timeout=10,
        # Temporarily allow all origins for debugging:
        origins=None # Or you might try ["*"] if None doesn't work
        # origins=["http://localhost:3000"] # Comment out the specific one for now
    )
    
    logger.info(f"WebSocket server started on {host}:{port}")
    
    # Start database listener
    db_listener_task = asyncio.create_task(db_listener(websocket_server))
    
    # Keep the server running indefinitely until cancelled (e.g., by Ctrl+C)
    # Signal handling removed as it's not reliable on Windows
    stop_event = asyncio.Event()
    try:
        await stop_event.wait() # Wait forever until cancelled
    except asyncio.CancelledError:
        logger.info("Server task cancelled.")
    finally:
        logger.info("Initiating server shutdown...")
        # Gracefully shutdown
        await shutdown(websocket_server, db_listener_task)

async def shutdown(websocket_server, db_listener_task):
    """Gracefully shut down the server."""
    logger.info("Shutting down server...")
    
    # Cancel database listener
    db_listener_task.cancel()
    try:
        await db_listener_task
    except asyncio.CancelledError:
        pass
    
    # Close all WebSocket connections
    for channel, clients in connections.items():
        for websocket in list(clients):
            await websocket.close(1001, "Server shutting down")
        clients.clear()
    
    # Close WebSocket server
    websocket_server.close()
    await websocket_server.wait_closed()
    
    # Stop event loop
    asyncio.get_event_loop().stop()
    
    logger.info("Server shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket server for real-time trading metrics")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user") 