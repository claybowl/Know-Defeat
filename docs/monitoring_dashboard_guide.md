# Real-Time Monitoring Dashboard Guide

This guide explains how to set up and run the real-time monitoring dashboard for the Know-Defeat Trading Platform.

## Overview

The monitoring dashboard consists of:

1. **Backend API** - FastAPI endpoints for retrieving metrics and status data
2. **Database Notification System** - PostgreSQL triggers for real-time data changes
3. **WebSocket Server** - For pushing real-time updates to clients
4. **Frontend Dashboard** - React-based UI for visualizing trading metrics

## Prerequisites

- PostgreSQL database running
- Python 3.8+ with conda environment activated
- Node.js 14+ for frontend development
- Database connection configured in `src/db_connection.py`

## Setup Instructions

### 1. Set Up Database Notifications

First, set up the PostgreSQL database triggers required for real-time notifications:

```bash
# Activate the Autogen conda environment
conda activate Autogen

# Start PostgreSQL if not already running
pg_ctl -D "C:/Users/clayb/postgres_data" start

# Run the notification setup script
python -m src.db.notifications_setup --setup
```

This creates the necessary triggers and functions in PostgreSQL to send notifications when data changes.

### 2. Run Database Indexes Creation

Optimize the database for monitoring queries:

```bash
# Connect to the database
psql -U clayb -d tick_data

# Run the following SQL commands
CREATE INDEX IF NOT EXISTS idx_bot_metrics_timestamp ON bot_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bot_metrics_bot_id_timestamp ON bot_metrics(bot_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON sim_bot_trades(trade_status);
CREATE INDEX IF NOT EXISTS idx_bot_metrics_ticker ON bot_metrics(ticker);
CREATE INDEX IF NOT EXISTS idx_bot_metrics_algorithm_type ON bot_metrics(algorithm_type);
```

These indexes will ensure the monitoring dashboard queries perform well even with large datasets.

### 3. Start the Main Application

The main application integrates the performance poller, API endpoints, and WebSocket server:

```bash
# Run the application
python -m src.main
```

This starts the FastAPI application on http://localhost:8000 with:
- REST API endpoints at `/api/...`
- API documentation at `/docs`
- WebSocket server for real-time updates
- Performance polling system in the background

### 4. Start the Frontend Development Server

To run the frontend dashboard in development mode:

```bash
# Navigate to the user_interface directory
cd user_interface

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The dashboard will be available at http://localhost:3000

### 5. Test the WebSocket Connection

To verify the WebSocket server is working properly:

```bash
# In a separate terminal, run the notification listener test
python -m src.db.notifications_setup --listen
```

Update data in the database and you should see notifications appear in the console.

## Dashboard Components

The dashboard includes these main components:

1. **Bot Metrics Panel** - Real-time metrics for all trading bots
2. **Active Trades Table** - Currently active trades with real-time P&L
3. **Bot Rankings** - Performance-based ranking of all bots
4. **Performance Charts** - Visual representation of key metrics
5. **System Status** - Health monitoring for all components

## API Endpoints

The following API endpoints are available:

- `GET /api/metrics/live` - Latest metrics from the bot_metrics table
- `GET /api/trades/active` - Currently active trades with real-time P&L
- `GET /api/bots/rankings` - Current bot rankings with performance scores
- `GET /api/system/heartbeat` - System status information
- `WebSocket /ws/metrics` - Real-time metrics updates

## WebSocket Channels

The WebSocket server provides these channels:

- `/bot_metrics` - Updates to bot metrics
- `/trades` - Updates to trade data
- `/rankings` - Updates to bot rankings

Connect to these channels using:

```javascript
const ws = new WebSocket('ws://localhost:8765/bot_metrics');
```

## Troubleshooting

### Database Connection Issues

If the dashboard can't connect to the database:

1. Ensure PostgreSQL is running
2. Verify database credentials in `src/db_connection.py`
3. Check database logs for any errors

### WebSocket Connection Issues

If real-time updates aren't working:

1. Verify the WebSocket server is running with `netstat -an | findstr 8765`
2. Check browser console for WebSocket errors
3. Ensure database triggers are correctly installed

### Performance Issues

If the dashboard is slow:

1. Verify the database indexes are created
2. Check the polling interval in `src/main.py` and adjust if needed
3. Consider adding a Redis cache layer as described in the architecture document

## Production Deployment

For production deployment:

1. Build the frontend:
   ```bash
   cd user_interface
   npm run build
   ```

2. Configure the FastAPI application to serve the frontend:
   - Uncomment the static file mounting in `src/main.py`
   - Set `allow_origins` to your specific domain
   - Use a production ASGI server like Gunicorn

3. Consider setting up a dedicated read replica database for monitoring queries

4. Use a process manager like systemd or supervisord to keep the server running

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Chakra UI Components](https://chakra-ui.com/docs/components)
- [PostgreSQL Notification Guide](https://www.postgresql.org/docs/current/sql-notify.html)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API) 