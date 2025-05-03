# Monitoring Components

This directory contains components for the real-time monitoring dashboard of the Know-Defeat algorithmic trading system.

## ActiveTradesTable Component

The `ActiveTradesTable` component displays currently active trades from all trading bots in real-time. It receives WebSocket data through the `/trades` channel and updates the UI accordingly.

### WebSocket Data Format

The component handles the following WebSocket message formats:

1. **Trade Opened**
   ```json
   {
     "channel": "trades",
     "data": {
       "action": "trade_opened",
       "trade_id": 12345,
       "bot_id": 1,
       "ticker": "TSLA",
       "entry_price": 250.45,
       "trade_size": 1000.0,
       "trade_direction": "LONG",
       "entry_time": "2023-04-20T12:34:56.789Z",
       "trade_status": "open",
       "trailing_stop_price": 247.95
     },
     "timestamp": 1681998896.789
   }
   ```

2. **Trade Closed**
   ```json
   {
     "channel": "trades",
     "data": {
       "action": "trade_closed",
       "trade_id": 12345,
       "bot_id": 1,
       "ticker": "TSLA",
       "exit_price": 255.75,
       "pnl": 5300.0,
       "pnl_percent": 2.12
     },
     "timestamp": 1681999996.789
   }
   ```

3. **Bulk Update**
   ```json
   {
     "channel": "trades",
     "data": {
       "action": "trade_update",
       "trades": [
         {
           "trade_id": 12345,
           "bot_id": 1,
           "ticker": "TSLA",
           "entry_price": 250.45,
           "trade_size": 1000.0,
           "trade_direction": "LONG",
           "entry_time": "2023-04-20T12:34:56.789Z",
           "trade_status": "open",
           "trailing_stop_price": 247.95
         },
         {
           "trade_id": 12346,
           "bot_id": 2,
           "ticker": "COIN",
           "entry_price": 155.20,
           "trade_size": 1000.0,
           "trade_direction": "SHORT",
           "entry_time": "2023-04-20T13:45:12.345Z",
           "trade_status": "open",
           "trailing_stop_price": 156.75
         }
       ],
       "timestamp": "2023-04-20T14:00:00.000Z"
     },
     "timestamp": 1682001600.000
   }
   ```

## Testing with Simulated Trades

You can test the `ActiveTradesTable` component using the provided test script:

```bash
# Interactive mode
python tests/test_websocket_trade_updates.py

# Automatic mode (random trades for 2 minutes with 5-second intervals)
python tests/test_websocket_trade_updates.py --auto --duration 120 --interval 5
```

### Test Script Options

The test script supports:
- Creating new trades (`Option 1: Send random trade_opened notification`)
- Closing trades (`Option 2: Send random trade_closed notification`)
- Sending bulk updates (`Option 3: Send bulk update with multiple active trades`)

### Prerequisites for Testing

1. Start the WebSocket server:
   ```bash
   python -m src.websocket_server
   ```

2. Start the trading UI:
   ```bash
   npm run dev
   ```

3. Navigate to the monitoring dashboard and select the "Active Trades" tab.

4. Run the test script to simulate trade updates.

## Troubleshooting WebSocket Connection

If you experience issues with the WebSocket connection:

1. Check the WebSocket server logs for errors:
   ```bash
   python -m src.websocket_server --debug
   ```

2. Verify PostgreSQL notification channel is working properly:
   ```bash
   psql -U clayb -d tick_data
   LISTEN trade_channel;
   -- In another terminal, run the test script
   ```

3. Check browser console for WebSocket connection errors.

4. Verify the WebSocket URL in `MonitoringDashboard.jsx` matches your server address:
   ```javascript
   const tradesWs = useWebSocket('ws://localhost:8765/trades');
   ``` 