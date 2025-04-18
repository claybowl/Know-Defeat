# Real-Time Monitoring Dashboard for Know Defeat Trading Platform

Based on the polling system implementation and existing architecture, this prompt outlines how to create a comprehensive real-time monitoring dashboard for the trading platform.

## Backend API Endpoints

First, establish the necessary API endpoints for the front end:

- **GET /api/metrics/live**
  - Returns the latest metrics from the `bot_metrics` table
  - Supports query parameters for filtering by `bot_id`, `ticker`, or `algorithm_type`

- **GET /api/trades/active**
  - Returns all currently active trades with real-time P&L calculations
  - Supports filtering by performance thresholds

- **GET /api/bots/rankings**
  - Returns the current bot rankings with performance scores
  - Includes activation status and historical rank changes

- **GET /api/system/heartbeat**
  - Returns system status information including database connectivity and polling system health

- **WebSocket Endpoint for Real-Time Updates**
  - `/ws/metrics` - Pushes updates whenever the `bot_metrics` table changes

## Front-End Implementation

The monitoring dashboard should include these key components:

### 1. Real-Time Bot Metrics Panel

```javascript
// React component using WebSockets for real-time updates
const BotMetricsPanel = () => {
  const [metrics, setMetrics] = useState([]);
  
  useEffect(() => {
    // Initial data load
    fetchMetrics();
    
    // Set up WebSocket connection
    const ws = new WebSocket('ws://yourserver.com/ws/metrics');
    
    ws.onmessage = (event) => {
      const newMetrics = JSON.parse(event.data);
      setMetrics(newMetrics);
    };
    
    return () => ws.close();
  }, []);
  
  const fetchMetrics = async () => {
    const response = await fetch('/api/metrics/live');
    const data = await response.json();
    setMetrics(data);
  };
  
  return (
    <div className="metrics-panel">
      <h2>Live Bot Performance</h2>
      <MetricsTable data={metrics} />
    </div>
  );
};
```

### 2. Bot Performance Dashboard

Create a dashboard that displays:
- Win rate trends (real-time)
- P&L by bot (updating in real-time)
- Performance heat map by algorithm type and ticker
- Active trade execution visualization
- Rank changes over time

### 3. Polling System Configuration Panel

Add a management interface for your polling system:
- Configure polling intervals
- Enable/disable specific metrics tracking
- Set alerting thresholds
- View polling system logs

### 4. Implementation Strategy

- Use server-sent events (SSE) or WebSockets for pushing real-time database changes to the client
- Implement a Redis cache layer to reduce database load for frequently accessed metrics
- Create a backend service that listens for database change notifications (PostgreSQL LISTEN/NOTIFY)
- Use React for the front-end with visualization libraries like Recharts or D3.js

## Sample Database Notification System

To capture real-time changes to the `bot_metrics` table, implement this PostgreSQL trigger:

```sql
CREATE OR REPLACE FUNCTION notify_bot_metrics_change()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('bot_metrics_channel', row_to_json(NEW)::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER bot_metrics_notify_trigger
AFTER INSERT OR UPDATE ON bot_metrics
FOR EACH ROW EXECUTE PROCEDURE notify_bot_metrics_change();
```

Then create a Node.js service that listens for these notifications and forwards them to connected WebSocket clients:

```javascript
const { Pool } = require('pg');
const WebSocket = require('ws');

const pool = new Pool(/* config */);
const wss = new WebSocket.Server({ port: 8080 });

// Set up PostgreSQL notification listener
pool.connect().then(client => {
  client.query('LISTEN bot_metrics_channel');
  
  client.on('notification', msg => {
    const payload = JSON.parse(msg.payload);
    
    // Forward to all connected WebSocket clients
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(payload));
      }
    });
  });
});
```

## Database Optimization

To ensure your real-time monitoring doesn't impact trading performance:
- Create dedicated read replicas for the monitoring dashboard
- Add indexes optimized for the specific queries used by the monitoring system:

```sql
-- Add indexes for common filtering and sorting operations
CREATE INDEX idx_bot_metrics_timestamp ON bot_metrics(timestamp DESC);
CREATE INDEX idx_bot_metrics_bot_id_timestamp ON bot_metrics(bot_id, timestamp DESC);
CREATE INDEX idx_trades_status ON sim_bot_trades(trade_status);
```

## Implementation Timeline

- **Week 1**: Set up database triggers and notification system
- **Week 2**: Develop API endpoints and WebSocket server
- **Week 3**: Create front-end dashboard components
- **Week 4**: Integrate and test the complete system

This approach will give you a comprehensive real-time monitoring solution that integrates with your existing architecture while minimizing changes to the core trading system. 