DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "tick_data",
    "user": "clayb",
    "password": "musicman"  # Use environment variable in real production
}

# Adjust polling frequencies for production
METRICS_POLLING_INTERVAL = 5  # seconds
WEBSOCKET_PING_INTERVAL = 30  # seconds

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/production.log"
