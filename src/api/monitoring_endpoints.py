import logging
import json
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from typing import List, Optional, Dict, Any
import asyncpg
from datetime import datetime
import asyncio

from src.db_connection import get_db_connection

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["monitoring"])

# WebSocket connections store
active_connections: List[WebSocket] = []

# Store for notification listeners
notification_listeners: List[asyncio.Task] = []

@router.get("/metrics/live")
async def get_live_metrics(
    bot_id: Optional[int] = Query(None),
    ticker: Optional[str] = Query(None),
    algo_id: Optional[int] = Query(None),
    connection=Depends(get_db_connection)
):
    """Get the latest metrics from the bot_metrics table with optional filtering."""
    try:
        # Build the query with optional filters
        query = """
            SELECT DISTINCT ON (bot_id) *
            FROM bot_metrics
        """
        
        conditions = []
        params = []
        param_index = 1
        
        if bot_id is not None:
            conditions.append(f"bot_id = ${param_index}")
            params.append(bot_id)
            param_index += 1
            
        if ticker is not None:
            conditions.append(f"ticker = ${param_index}")
            params.append(ticker)
            param_index += 1
            
        if algo_id is not None:
            conditions.append(f"algo_id = ${param_index}")
            params.append(algo_id)
            param_index += 1
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY bot_id, timestamp DESC"
        
        # Execute the query
        metrics = await connection.fetch(query, *params)
        
        # Convert to list of dicts for JSON response
        result = [dict(metric) for metric in metrics]
        
        return {"metrics": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Error getting live metrics: {e}")
        return {"error": str(e)}

@router.get("/trades/active")
async def get_active_trades(
    min_pnl: Optional[float] = Query(None),
    connection=Depends(get_db_connection)
):
    """Get all currently active trades with real-time P&L calculations."""
    try:
        # Build query with optional PnL filtering
        query = """
            SELECT t.trade_id, t.bot_id, b.name AS bot_name, t.ticker,
                   t.entry_price, t.trade_size, t.trade_direction, t.entry_time,
                   t.trailing_stop_price,
                   CASE 
                       WHEN t.trade_direction = 'LONG' THEN 
                           (SELECT price FROM tick_data WHERE ticker = t.ticker ORDER BY timestamp DESC LIMIT 1) - t.entry_price
                       ELSE 
                           t.entry_price - (SELECT price FROM tick_data WHERE ticker = t.ticker ORDER BY timestamp DESC LIMIT 1)
                   END * t.trade_size AS current_pnl
            FROM sim_bot_trades t
            JOIN sim_bots b ON t.bot_id = b.bot_id
            WHERE t.trade_status = 'open'
        """
        
        params = []
        if min_pnl is not None:
            query += " HAVING current_pnl >= $1"
            params.append(min_pnl)
            
        query += " ORDER BY current_pnl DESC"
        
        # Execute the query
        trades = await connection.fetch(query, *params)
        
        # Convert to list of dicts for JSON response
        result = [dict(trade) for trade in trades]
        
        return {"trades": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Error getting active trades: {e}")
        return {"error": str(e)}

@router.get("/bots/rankings")
async def get_bot_rankings(connection=Depends(get_db_connection)):
    """Get the current bot rankings with performance scores."""
    try:
        # Get current rankings
        rankings = await connection.fetch("""
            SELECT r.bot_id, r.rank_score, r.rank, r.is_active, 
                   b.name, b.ticker, b.algorithm_type,
                   (SELECT r2.rank FROM bot_rankings r2
                    WHERE r2.bot_id = r.bot_id
                    AND r2.timestamp < r.timestamp
                    ORDER BY r2.timestamp DESC LIMIT 1) AS previous_rank
            FROM bot_rankings r
            JOIN sim_bots b ON r.bot_id = b.bot_id
            ORDER BY r.rank
        """)
        
        # Calculate rank changes
        result = []
        for ranking in rankings:
            rank_dict = dict(ranking)
            
            # Calculate rank change if previous rank exists
            if ranking['previous_rank'] is not None:
                rank_dict['rank_change'] = ranking['previous_rank'] - ranking['rank']
            else:
                rank_dict['rank_change'] = 0
                
            result.append(rank_dict)
        
        return {"rankings": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Error getting bot rankings: {e}")
        return {"error": str(e)}

@router.get("/system/heartbeat")
async def get_system_heartbeat(connection=Depends(get_db_connection)):
    """Get system status information."""
    try:
        # Check database connection
        db_status = "connected"
        try:
            await connection.execute("SELECT 1")
        except Exception:
            db_status = "disconnected"
            
        # Get latest poll time from poller logs (if available)
        last_poll_time = None
        try:
            poll_record = await connection.fetchrow("""
                SELECT timestamp FROM system_logs
                WHERE action = 'poll_complete'
                ORDER BY timestamp DESC LIMIT 1
            """)
            if poll_record:
                last_poll_time = poll_record['timestamp']
        except Exception:
            # Table might not exist yet
            pass
        
        # Get active bot count
        active_bot_count = 0
        try:
            count_record = await connection.fetchrow("""
                SELECT COUNT(*) FROM sim_bots WHERE is_active = TRUE
            """)
            if count_record:
                active_bot_count = count_record['count']
        except Exception:
            # Handle case where table doesn't exist
            pass
            
        return {
            "timestamp": datetime.now().isoformat(),
            "database_status": db_status,
            "last_poll_time": last_poll_time,
            "active_bot_count": active_bot_count,
            "active_websocket_connections": len(active_connections)
        }
    except Exception as e:
        logger.error(f"Error getting system heartbeat: {e}")
        return {"error": str(e)}

@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket, connection=Depends(get_db_connection)):
    """WebSocket endpoint for real-time metrics updates using PostgreSQL LISTEN/NOTIFY."""
    await websocket.accept()
    active_connections.append(websocket)
    
    # Create a direct connection for LISTEN/NOTIFY
    # We need a separate connection because the pooled connection
    # might be shared between requests
    try:
        # Create a dedicated connection for notifications
        notify_conn = await asyncpg.connect(
            user='clayb',
            password='musicman',
            database='tick_data',
            host='localhost'
        )
        
        # Send initial data
        metrics = await connection.fetch("""
            SELECT DISTINCT ON (bot_id) *
            FROM bot_metrics
            ORDER BY bot_id, timestamp DESC
        """)
        
        await websocket.send_text(json.dumps([dict(metric) for metric in metrics]))
        
        # Set up LISTEN on the notification channel
        await notify_conn.execute("LISTEN bot_metrics_channel")
        
        # Create a task to handle notifications
        async def listen_for_notifications():
            while True:
                try:
                    # Wait for notifications
                    notification = await notify_conn.fetchrow(
                        "SELECT 1 as dummy_col"
                    )
                    
                    # Process notifications from the connection
                    notifications = notify_conn.notifications
                    if notifications:
                        for notification in notifications:
                            # Parse the notification payload
                            payload = json.loads(notification.payload)
                            
                            # Get updated metrics for the affected bot
                            if payload.get('table') == 'bot_metrics':
                                bot_id = payload.get('data', {}).get('bot_id')
                                if bot_id:
                                    # Fetch the latest metrics for this bot
                                    updated_metrics = await connection.fetch("""
                                        SELECT * FROM bot_metrics
                                        WHERE bot_id = $1
                                        ORDER BY timestamp DESC
                                        LIMIT 1
                                    """, bot_id)
                                    
                                    if updated_metrics:
                                        # Send the update to the client
                                        await websocket.send_text(json.dumps([dict(metric) for metric in updated_metrics]))
                        
                        # Clear notifications
                        notify_conn.notifications.clear()
                        
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error processing notification: {e}")
                    await asyncio.sleep(1)  # Wait before retrying
        
        # Start the notification listener task
        notification_task = asyncio.create_task(listen_for_notifications())
        notification_listeners.append(notification_task)
        
        # Keep connection open until client disconnects
        while True:
            # This will raise WebSocketDisconnect when client disconnects
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up on disconnect
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        # Cancel notification listener task
        for task in notification_listeners:
            if not task.done():
                task.cancel()
        
        # Close notification connection
        try:
            await notify_conn.close()
        except Exception:
            pass

async def broadcast_metrics_update(metrics_data: Dict[str, Any]):
    """Broadcast metrics updates to all connected WebSocket clients."""
    if not active_connections:
        return
        
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(metrics_data))
        except Exception as e:
            logger.error(f"Error broadcasting to WebSocket: {e}") 