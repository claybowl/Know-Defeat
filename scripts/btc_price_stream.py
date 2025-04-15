#!/usr/bin/env python3
"""
BTC Price Data Stream using CoinMarketCap API

This script fetches BTC price data from CoinMarketCap API and stores it in PostgreSQL.
It runs as a continuous service that updates prices at a configurable interval.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timezone
from decimal import Decimal

# Handle import errors gracefully
try:
    import asyncpg
    import asyncio
    import requests
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install required packages using:")
    print("  pip install asyncpg requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("btc_price_stream.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "ee07822a-66e7-499f-909d-02821cf2d58a"  # Your CoinMarketCap API key
UPDATE_INTERVAL = 60  # Seconds between updates (1 minute)
SYMBOL = "BTC"  # Cryptocurrency symbol
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

# API Endpoints
LATEST_QUOTES_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

async def create_price_table(conn):
    """Create the BTC price table if it doesn't exist."""
    await conn.execute('''
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
    ''')
    
    # Create index on timestamp for faster queries
    await conn.execute('''
        CREATE INDEX IF NOT EXISTS btc_price_data_timestamp_idx ON btc_price_data (timestamp)
    ''')
    
    logger.info("BTC price table created or verified")

def fetch_btc_data():
    """Fetch latest BTC price data from CoinMarketCap API."""
    try:
        headers = {
            'X-CMC_PRO_API_KEY': API_KEY,
            'Accept': 'application/json'
        }
        
        params = {
            'symbol': SYMBOL,
            'convert': 'USD'
        }
        
        response = requests.get(LATEST_QUOTES_URL, headers=headers, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        data = response.json()
        
        if 'data' not in data or SYMBOL not in data['data']:
            logger.error(f"Unexpected API response format: {data}")
            return None
        
        btc_data = data['data'][SYMBOL]
        quote = btc_data['quote']['USD']
        
        result = {
            'timestamp': datetime.now(timezone.utc),
            'price': Decimal(str(quote['price'])),
            'volume_24h': Decimal(str(quote['volume_24h'])) if 'volume_24h' in quote else None,
            'market_cap': Decimal(str(quote['market_cap'])) if 'market_cap' in quote else None,
            'percent_change_1h': Decimal(str(quote['percent_change_1h'])) if 'percent_change_1h' in quote else None,
            'percent_change_24h': Decimal(str(quote['percent_change_24h'])) if 'percent_change_24h' in quote else None,
            'percent_change_7d': Decimal(str(quote['percent_change_7d'])) if 'percent_change_7d' in quote else None,
            'circulating_supply': Decimal(str(btc_data['circulating_supply'])) if 'circulating_supply' in btc_data else None
        }
        
        logger.info(f"Fetched BTC price: ${result['price']:,.2f}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Data parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching BTC data: {e}")
        return None

async def save_price_data(conn, price_data):
    """Save BTC price data to database."""
    try:
        await conn.execute('''
            INSERT INTO btc_price_data (
                timestamp, price, volume_24h, market_cap, 
                percent_change_1h, percent_change_24h, percent_change_7d,
                circulating_supply
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ''', price_data['timestamp'], price_data['price'], price_data['volume_24h'], 
        price_data['market_cap'], price_data['percent_change_1h'], 
        price_data['percent_change_24h'], price_data['percent_change_7d'],
        price_data['circulating_supply'])
        
        logger.info(f"Saved BTC price data to database")
        return True
    except Exception as e:
        logger.error(f"Database error saving price data: {e}")
        return False

async def run_price_stream():
    """Run the continuous BTC price data stream."""
    # Connect to database
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("Connected to database")
        
        # Create table if it doesn't exist
        await create_price_table(conn)
        
        # Main loop to fetch and store data
        while True:
            try:
                # Fetch BTC price data
                price_data = fetch_btc_data()
                
                if price_data:
                    # Save to database
                    await save_price_data(conn, price_data)
                
                # Wait for next update
                await asyncio.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in price stream loop: {e}")
                await asyncio.sleep(5)  # Wait a bit before retrying
                
    except Exception as e:
        logger.error(f"Database connection error: {e}")
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("Database connection closed")

async def query_latest_prices(num_records=10):
    """Query the latest BTC prices from the database."""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        rows = await conn.fetch('''
            SELECT 
                timestamp, 
                price, 
                percent_change_1h,
                percent_change_24h
            FROM btc_price_data
            ORDER BY timestamp DESC
            LIMIT $1
        ''', num_records)
        
        if not rows:
            print("No BTC price data found in database.")
            return
        
        print("\nLatest BTC Prices:")
        print("=" * 60)
        print(f"{'Timestamp':<25} {'Price':<15} {'1h %':<10} {'24h %':<10}")
        print("-" * 60)
        
        for row in rows:
            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            price = f"${float(row['price']):,.2f}"
            pct_1h = f"{float(row['percent_change_1h'] or 0):.2f}%" if row['percent_change_1h'] else "N/A"
            pct_24h = f"{float(row['percent_change_24h'] or 0):.2f}%" if row['percent_change_24h'] else "N/A"
            
            print(f"{timestamp:<25} {price:<15} {pct_1h:<10} {pct_24h:<10}")
        
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        if 'conn' in locals():
            await conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BTC Price Data Stream')
    parser.add_argument('--query', action='store_true', help='Query latest prices instead of running the stream')
    parser.add_argument('--count', type=int, default=10, help='Number of records to show when querying')
    
    args = parser.parse_args()
    
    if args.query:
        asyncio.run(query_latest_prices(args.count))
    else:
        logger.info("Starting BTC price data stream")
        try:
            asyncio.run(run_price_stream())
        except KeyboardInterrupt:
            logger.info("BTC price stream stopped by user")