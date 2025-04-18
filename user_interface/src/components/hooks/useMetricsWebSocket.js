import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom React hook for connecting to the bot metrics WebSocket
 * Handles connection, reconnection, and data updates
 * 
 * @param {string} url - The WebSocket URL (default: 'ws://localhost:8000/api/ws/metrics')
 * @param {Object} options - Configuration options
 * @param {boolean} options.autoConnect - Whether to connect automatically (default: true)
 * @param {number} options.reconnectInterval - Milliseconds to wait before reconnecting (default: 2000)
 * @param {number} options.maxReconnectAttempts - Maximum number of reconnect attempts (default: 5)
 * @returns {Object} WebSocket state and control functions
 */
const useMetricsWebSocket = (
  url = 'ws://localhost:8000/api/ws/metrics',
  options = {}
) => {
  // Default options
  const {
    autoConnect = true,
    reconnectInterval = 2000,
    maxReconnectAttempts = 5
  } = options;

  // State
  const [isConnected, setIsConnected] = useState(false);
  const [metrics, setMetrics] = useState([]);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  // Refs
  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // Function to connect to the WebSocket
  const connect = useCallback(() => {
    // Clear any existing connection
    if (ws.current) {
      ws.current.close();
    }

    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      // Create new WebSocket connection
      ws.current = new WebSocket(url);

      // WebSocket event handlers
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        setReconnectAttempts(0);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          
          // Update metrics - handle both array and object responses
          if (Array.isArray(data)) {
            // Initial data load or full refresh
            setMetrics(data);
          } else if (data.metrics && Array.isArray(data.metrics)) {
            // API-style response with metrics array
            setMetrics(data.metrics);
          } else {
            // Single metric update - merge with existing metrics
            setMetrics(prevMetrics => {
              // Create a map of current metrics by bot_id
              const metricsMap = new Map(
                prevMetrics.map(metric => [metric.bot_id, metric])
              );
              
              // Update metrics with new data
              data.forEach(update => {
                if (update.bot_id) {
                  metricsMap.set(update.bot_id, {
                    ...metricsMap.get(update.bot_id),
                    ...update
                  });
                }
              });
              
              // Convert back to array
              return Array.from(metricsMap.values());
            });
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.current.onclose = (event) => {
        setIsConnected(false);
        console.log('WebSocket disconnected, code:', event.code);
        
        // Attempt to reconnect unless explicitly closed
        if (event.code !== 1000) {
          scheduleReconnect();
        }
      };

      ws.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
      };
    } catch (err) {
      setError(`Failed to connect: ${err.message}`);
      scheduleReconnect();
    }
  }, [url]);

  // Function to schedule a reconnection attempt
  const scheduleReconnect = useCallback(() => {
    if (reconnectAttempts < maxReconnectAttempts) {
      console.log(`Scheduling reconnect attempt ${reconnectAttempts + 1}/${maxReconnectAttempts}`);
      reconnectTimeoutRef.current = setTimeout(() => {
        setReconnectAttempts(prev => prev + 1);
        connect();
      }, reconnectInterval);
    } else {
      setError(`Maximum reconnect attempts (${maxReconnectAttempts}) exceeded`);
    }
  }, [reconnectAttempts, maxReconnectAttempts, reconnectInterval, connect]);

  // Function to manually disconnect
  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close(1000, 'Closed by user');
      ws.current = null;
    }
    
    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  // Connect on mount if autoConnect is true
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close(1000, 'Component unmounted');
        ws.current = null;
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [autoConnect, connect]);

  // Return the WebSocket state and control functions
  return {
    isConnected,
    metrics,
    lastMessage,
    error,
    connect,
    disconnect,
    reconnectAttempts
  };
};

export default useMetricsWebSocket; 