# Real-Time WebSocket Implementation for Bot Metrics

This document explains how to set up and use the real-time WebSocket implementation for bot metrics in the Know-Defeat trading system.

## Overview

The system uses PostgreSQL's LISTEN/NOTIFY feature to provide real-time updates when bot metrics change. When a bot metric is updated in the database, a notification is sent to connected WebSocket clients.

## Components

1. **PostgreSQL Trigger**: Sends notifications when bot metrics are updated
2. **WebSocket Endpoint**: `/api/ws/metrics` - Accepts client connections and forwards notifications
3. **Database Listener**: Listens for PostgreSQL notifications and processes them
4. **Client Integration**: JavaScript code to connect to the WebSocket and handle updates

## Setup Instructions

### 1. Apply the PostgreSQL Trigger

Run the `apply_notification_trigger.py` script to set up the database trigger:

```bash
conda activate Autogen
python scripts/apply_notification_trigger.py
```

This script:
- Creates a notification function that sends JSON payloads when bot metrics change
- Sets up a trigger on the `bot_metrics` table for INSERT and UPDATE operations
- Tests the trigger to verify it's working

### 2. Start the FastAPI Server

Make sure your FastAPI server is running with WebSocket support:

```bash
conda activate Autogen
uvicorn src.main:app --reload
```

### 3. Test the WebSocket Connection

Run the test script to verify the WebSocket implementation is working correctly:

```bash
conda activate Autogen
python tests/test_metrics_websocket.py
```

This script:
- Connects to the WebSocket endpoint
- Listens for messages
- Updates a bot metric in the database
- Verifies that an update notification is received

## Client Integration

To integrate this with a frontend UI, add the following JavaScript code:

```javascript
// Initialize WebSocket connection
function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8000/api/ws/metrics');
  
  ws.onopen = function() {
    console.log('WebSocket connected');
  };
  
  ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received metrics update:', data);
    
    // Update UI with the new data
    updateMetricsUI(data);
  };
  
  ws.onclose = function() {
    console.log('WebSocket disconnected');
    // Attempt to reconnect after a delay
    setTimeout(connectWebSocket, 2000);
  };
  
  ws.onerror = function(error) {
    console.error('WebSocket error:', error);
    ws.close();
  };
  
  return ws;
}

// Function to update UI with new metrics data
function updateMetricsUI(metricsData) {
  // Example implementation - update as needed for your UI
  for (const metric of metricsData) {
    const botId = metric.bot_id;
    
    // Find the corresponding UI element
    const element = document.querySelector(`[data-bot-id="${botId}"]`);
    if (element) {
      // Update metrics values
      element.querySelector('.win-rate').textContent = 
        (parseFloat(metric.win_rate) * 100).toFixed(2) + '%';
      element.querySelector('.total-pnl').textContent = 
        '$' + parseFloat(metric.total_pnl).toFixed(2);
      // Update other metrics as needed
    }
  }
}

// Connect when the page loads
document.addEventListener('DOMContentLoaded', function() {
  const ws = connectWebSocket();
  
  // Store the WebSocket reference for cleanup
  window.metricsWebSocket = ws;
  
  // Clean up on page unload
  window.addEventListener('beforeunload', function() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  });
});
```

## How It Works

1. **Database Updates**:
   - When a bot metric is updated in the `bot_metrics` table
   - The PostgreSQL trigger fires and sends a notification via `pg_notify`
   - The notification includes the bot ID and updated data

2. **Server Processing**:
   - The FastAPI WebSocket endpoint listens for these notifications
   - When a notification is received, the server:
     - Fetches the latest data for the affected bot
     - Sends the updated data to all connected WebSocket clients

3. **Client Updates**:
   - The client receives the updated data via WebSocket
   - The client updates the UI to reflect the changes in real-time

## Troubleshooting

### WebSocket Connection Issues

- Verify the FastAPI server is running with WebSocket support
- Check for any CORS issues if connecting from a different domain
- Ensure the WebSocket URL is correct (ws://localhost:8000/api/ws/metrics)

### Missing Notifications

- Verify the PostgreSQL trigger is installed correctly
- Check PostgreSQL logs for any errors related to the notification function
- Run `SELECT pg_listening_channels();` in PostgreSQL to verify the channel is being listened to

### Performance Considerations

- For high-volume updates, consider batching notifications
- WebSocket connections use server resources, so limit the number of connections per client
- Consider implementing a message queue for larger deployments 