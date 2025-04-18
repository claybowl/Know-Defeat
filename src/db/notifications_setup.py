import asyncio
import logging
import argparse
from src.db_connection import create_db_pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# SQL to create notification function and trigger
NOTIFICATION_SETUP_SQL = """
-- Create notification function for bot_metrics changes
CREATE OR REPLACE FUNCTION notify_bot_metrics_change()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('bot_metrics_channel', row_to_json(NEW)::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for bot_metrics table
DROP TRIGGER IF EXISTS bot_metrics_notify_trigger ON bot_metrics;
CREATE TRIGGER bot_metrics_notify_trigger
AFTER INSERT OR UPDATE ON bot_metrics
FOR EACH ROW EXECUTE PROCEDURE notify_bot_metrics_change();

-- Create notification function for active trades
CREATE OR REPLACE FUNCTION notify_trade_change()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('trade_channel', row_to_json(NEW)::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for sim_bot_trades table
DROP TRIGGER IF EXISTS trade_notify_trigger ON sim_bot_trades;
CREATE TRIGGER trade_notify_trigger
AFTER INSERT OR UPDATE ON sim_bot_trades
FOR EACH ROW EXECUTE PROCEDURE notify_trade_change();

-- Create notification function for bot rankings
CREATE OR REPLACE FUNCTION notify_ranking_change()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('ranking_channel', row_to_json(NEW)::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for bot_rankings table
DROP TRIGGER IF EXISTS ranking_notify_trigger ON bot_rankings;
CREATE TRIGGER ranking_notify_trigger
AFTER INSERT OR UPDATE ON bot_rankings
FOR EACH ROW EXECUTE PROCEDURE notify_ranking_change();
"""

# SQL to create database indexes for monitoring
INDEX_SETUP_SQL = """
-- Add indexes for common filtering and sorting operations
CREATE INDEX IF NOT EXISTS idx_bot_metrics_timestamp ON bot_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bot_metrics_bot_id_timestamp ON bot_metrics(bot_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON sim_bot_trades(trade_status);
CREATE INDEX IF NOT EXISTS idx_bot_metrics_ticker ON bot_metrics(ticker);
-- CREATE INDEX IF NOT EXISTS idx_bot_metrics_algorithm_type ON bot_metrics(algorithm_type);
"""

async def setup_notifications():
    """Set up PostgreSQL notification triggers and indexes."""
    pool = await create_db_pool()
    try:
        async with pool.acquire() as conn:
            # Execute the notification setup SQL
            logger.info("Setting up notification triggers...")
            await conn.execute(NOTIFICATION_SETUP_SQL)
            
            # Create indexes
            logger.info("Creating database indexes...")
            await conn.execute(INDEX_SETUP_SQL)
            
            logger.info("Database notification system and indexes set up successfully")
    except Exception as e:
        logger.error(f"Error setting up notification system: {e}")
    finally:
        await pool.close()

async def listen_for_notifications():
    """Test listening for database notifications."""
    pool = await create_db_pool()
    try:
        # Get a connection for listening
        connection = await pool.acquire()
        
        # Listen for notifications
        await connection.add_listener('bot_metrics_channel', on_notification)
        await connection.add_listener('trade_channel', on_notification)
        await connection.add_listener('ranking_channel', on_notification)
        
        logger.info("Listening for database notifications. Press Ctrl+C to stop.")
        
        # Keep the script running
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Notification listener stopped")
    except Exception as e:
        logger.error(f"Error in notification listener: {e}")
    finally:
        # Remove listeners and release connection
        await connection.remove_listener('bot_metrics_channel', on_notification)
        await connection.remove_listener('trade_channel', on_notification)
        await connection.remove_listener('ranking_channel', on_notification)
        await pool.release(connection)
        await pool.close()

async def on_notification(connection, pid, channel, payload):
    """Handle received notifications."""
    logger.info(f"Notification on channel {channel}: {payload[:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up and test PostgreSQL notification system")
    parser.add_argument("--setup", action="store_true", help="Set up notification triggers and indexes")
    parser.add_argument("--listen", action="store_true", help="Test listening for notifications")
    
    args = parser.parse_args()
    
    if args.setup:
        asyncio.run(setup_notifications())
    elif args.listen:
        try:
            asyncio.run(listen_for_notifications())
        except KeyboardInterrupt:
            logger.info("Listener stopped by user")
    else:
        parser.print_help() 