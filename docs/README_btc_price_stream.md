# BTC Price Stream - User Guide

This tool fetches Bitcoin price data from CoinMarketCap API and stores it in your PostgreSQL database. It can run as a continuous background service or be used to query previously collected data.

## Setup Instructions

### 1. Install Dependencies

#### For Windows:
```
scripts\install_btc_deps_win.bat
```

#### For Linux/Mac:
```
./scripts/install_btc_deps.sh
```

### 2. Using the BTC Price Stream

#### Query Latest Prices

To view the most recent Bitcoin prices stored in the database:

##### Windows:
```
scripts\query_btc_prices.bat [number_of_records]
```

##### Linux/Mac:
```
./scripts/query_btc_prices.sh [number_of_records]
```

The default is to show the 10 most recent records.

#### Start the Continuous Data Stream

To start collecting Bitcoin price data in the background:

##### Windows:
```
scripts\start_btc_stream.bat
```

##### Linux/Mac:
```
./scripts/start_btc_stream.sh
```

This will run the price collection service in the background, saving data to the database every minute.

#### Stop the Data Stream

To stop the running data collection service:

##### Windows:
```
scripts\stop_btc_stream.bat
```

##### Linux/Mac:
```
./scripts/stop_btc_stream.sh
```

## Configuration

You can modify these settings in the `btc_price_stream.py` file:

- **API_KEY**: Your CoinMarketCap API key
- **UPDATE_INTERVAL**: How often to fetch prices (in seconds)
- **DB_CONFIG**: Database connection settings

## Logs

Logs are written to `btc_price_stream.log` in the current directory.

## Database Schema

The script creates a table called `btc_price_data` with the following structure:

```sql
CREATE TABLE IF NOT EXISTS btc_price_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume_24h DECIMAL(24, 8),
    market_cap DECIMAL(24, 8),
    percent_change_1h DECIMAL(10, 4),
    percent_change_24h DECIMAL(10, 4),
    percent_change_7d DECIMAL(10, 4),
    circulating_supply DECIMAL(24, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
)
```

## Troubleshooting

If you encounter any issues:

1. **Missing Python packages**: Run the appropriate install script mentioned above
2. **Database connection errors**: Verify your PostgreSQL server is running and the credentials in the script are correct
3. **API errors**: Check your API key and internet connection