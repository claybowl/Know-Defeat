# PostgreSQL Database Schema for Know-Defeat Trading System

This document provides comprehensive information about the database tables, relationships, and design patterns used in the Know-Defeat algorithmic trading system.

## Database Connection Details

```python
# Connection parameters
host = 'localhost'
port = 5432
database = 'tick_data'
user = 'clayb'
password = 'musicman'

# Connection string format
connection_string = f"postgres://{user}:{password}@{host}:{port}/{database}"

# Using asyncpg (recommended for async operations)
pool = await asyncpg.create_pool(
    user=user,
    password=password,
    database=database,
    host=host,
    port=port,
    min_size=5,
    max_size=20
)

# Using psycopg2 (for synchronous operations)
conn = psycopg2.connect(
    dbname=database,
    user=user,
    password=password,
    host=host,
    port=port
)
```

## Table Schemas

### `tick_data` Table

Stores market data at the tick level for all tracked securities.

```sql
CREATE TABLE tick_data (
    timestamp TIMESTAMP NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    price DECIMAL(15,6) NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (timestamp, ticker)
);

-- Create indexes for performance
CREATE INDEX idx_tick_data_ticker ON tick_data(ticker);
CREATE INDEX idx_tick_data_timestamp ON tick_data(timestamp);
```

### `sim_bots` Table

Stores configuration details for all trading bots in the system.

```sql
CREATE TABLE sim_bots (
    bot_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    algorithm_module VARCHAR(255) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    trade_direction VARCHAR(10) NOT NULL,
    position_size NUMERIC(15,2) NOT NULL,
    trailing_stop_pct NUMERIC(8,6) NOT NULL,
    description TEXT,
    version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX idx_sim_bots_ticker ON sim_bots(ticker);
CREATE INDEX idx_sim_bots_active ON sim_bots(is_active);
```

### `sim_bot_trades` Table

Records all trades made by the simulation bots.

```sql
CREATE TABLE sim_bot_trades (
    trade_id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL REFERENCES sim_bots(bot_id),
    ticker VARCHAR(10) NOT NULL,
    entry_price NUMERIC(15,6) NOT NULL,
    exit_price NUMERIC(15,6),
    trade_size NUMERIC(15,2) NOT NULL,
    trade_direction VARCHAR(10) NOT NULL,
    entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMP,
    trade_status VARCHAR(20) NOT NULL DEFAULT 'open',
    pnl NUMERIC(15,2),
    pnl_percent NUMERIC(15,6),
    trailing_stop_price NUMERIC(15,6),
    exit_reason VARCHAR(50)
);

-- Create indexes for performance
CREATE INDEX idx_sim_bot_trades_bot_id ON sim_bot_trades(bot_id);
CREATE INDEX idx_sim_bot_trades_status ON sim_bot_trades(trade_status);
CREATE INDEX idx_sim_bot_trades_ticker ON sim_bot_trades(ticker);
CREATE INDEX idx_sim_bot_trades_entry_time ON sim_bot_trades(entry_time);
```

### `bot_tick_data` Table

Stores tick data specifically for bot consumption, with processing status tracking.

```sql
CREATE TABLE bot_tick_data (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    price NUMERIC(15,6) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_bot_id
        FOREIGN KEY(bot_id) 
        REFERENCES sim_bots(bot_id)
);

-- Create indexes for performance
CREATE INDEX idx_bot_tick_data_ticker ON bot_tick_data(ticker);
CREATE INDEX idx_bot_tick_data_bot_id ON bot_tick_data(bot_id);
CREATE INDEX idx_bot_tick_data_processed ON bot_tick_data(processed);
```

### `bot_metrics` Table

Stores performance metrics for each bot to facilitate ranking and analysis.

```sql
CREATE TABLE bot_metrics (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL REFERENCES sim_bots(bot_id),
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl NUMERIC(15,2) NOT NULL DEFAULT 0,
    average_pnl_per_trade NUMERIC(15,2) NOT NULL DEFAULT 0,
    win_rate NUMERIC(5,4) NOT NULL **DEFAULT** 0,
    average_win_amount NUMERIC(15,2) NOT NULL DEFAULT 0,
    average_loss_amount NUMERIC(15,2) NOT NULL DEFAULT 0,
    profit_factor NUMERIC(15,4) NOT NULL DEFAULT 0,
    max_drawdown NUMERIC(15,2) NOT NULL DEFAULT 0,
    sharpe_ratio NUMERIC(10,4) NOT NULL DEFAULT 0,
    risk_reward_ratio NUMERIC(10,4) NOT NULL DEFAULT 0,
    expectancy NUMERIC(10,4) NOT NULL DEFAULT 0,
    rank_score NUMERIC(10,4) NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index for bot_id for fast lookups
CREATE INDEX idx_bot_metrics_bot_id ON bot_metrics(bot_id);
```

### `variable_weights` Table

Stores weight configurations for the algorithmic ranking system.

```sql
CREATE TABLE variable_weights (
    variable_name VARCHAR(50) PRIMARY KEY,
    weight NUMERIC(5,4) NOT NULL DEFAULT 1.0,
    description TEXT,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## Common Query Patterns

### Fetching Recent Tick Data for a Symbol

```sql
SELECT timestamp, price, volume 
FROM tick_data
WHERE ticker = $1
  AND timestamp > NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC
LIMIT 1000;
```

### Getting Active Bot List

```sql
SELECT bot_id, name, ticker, algorithm_type, trade_direction
FROM sim_bots
WHERE is_active = TRUE
ORDER BY bot_id;
```

### Retrieving Open Trades

```sql
SELECT t.trade_id, t.bot_id, b.name AS bot_name, t.ticker, 
       t.entry_price, t.trade_size, t.trade_direction, 
       t.entry_time, t.trailing_stop_price
FROM sim_bot_trades t
JOIN sim_bots b ON t.bot_id = b.bot_id
WHERE t.trade_status = 'open'
ORDER BY t.entry_time DESC;
```

### Calculating Bot Performance

```sql
SELECT 
    b.bot_id,
    b.name,
    b.ticker,
    b.algorithm_type,
    COUNT(t.trade_id) AS total_trades,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS winning_trades,
    SUM(CASE WHEN t.pnl <= 0 THEN 1 ELSE 0 END) AS losing_trades,
    SUM(t.pnl) AS total_pnl,
    AVG(t.pnl) AS avg_pnl_per_trade,
    CASE 
        WHEN COUNT(t.trade_id) > 0 THEN 
            SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(t.trade_id) 
        ELSE 0 
    END AS win_rate
FROM sim_bots b
LEFT JOIN sim_bot_trades t ON b.bot_id = t.bot_id AND t.trade_status = 'closed'
WHERE b.is_active = TRUE
GROUP BY b.bot_id, b.name, b.ticker, b.algorithm_type
ORDER BY total_pnl DESC;
```

## Python Connection Examples

### Asyncpg (Recommended)

```python
import asyncio
import asyncpg

async def example_query():
    # Create a connection pool
    pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )
    
    # Use the pool to execute a query
    async with pool.acquire() as conn:
        # Fetch a single value
        count = await conn.fetchval('SELECT COUNT(*) FROM tick_data')
        print(f"Total tick records: {count}")
        
        # Fetch multiple rows
        rows = await conn.fetch(
            'SELECT ticker, COUNT(*) FROM tick_data GROUP BY ticker'
        )
        for row in rows:
            print(f"Ticker: {row['ticker']}, Count: {row['count']}")
    
    # Close the pool
    await pool.close()

# Run the async function
asyncio.run(example_query())
```

### Psycopg2 (Synchronous)

```python
import psycopg2
import psycopg2.extras

def example_query():
    # Create a connection
    conn = psycopg2.connect(
        dbname="tick_data",
        user="clayb",
        password="musicman",
        host="localhost"
    )
    
    # Create a cursor with dictionary results
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Execute a query
    cur.execute('SELECT COUNT(*) FROM tick_data')
    count = cur.fetchone()[0]
    print(f"Total tick records: {count}")
    
    # Execute another query
    cur.execute('SELECT ticker, COUNT(*) FROM tick_data GROUP BY ticker')
    for row in cur.fetchall():
        print(f"Ticker: {row['ticker']}, Count: {row['count']}")
    
    # Close connections
    cur.close()
    conn.close()

# Run the function
example_query()
```

## Best Practices for Database Operations

1. **Use connection pooling**: Always use connection pools for better performance and resource management.

2. **Parameterized queries**: Always use parameterized queries to prevent SQL injection.

3. **Transaction management**: Use transactions for operations that should be atomic.

4. **Batch operations**: For multiple inserts, use batch operations like `executemany()` or `copy_records_to_table()`.

5. **Index optimization**: Create appropriate indexes for common query patterns.

6. **Regular maintenance**: Implement regular VACUUM and ANALYZE operations to maintain database health.

7. **Error handling**: Implement proper error handling and retry mechanisms for database operations.

8. **Connection timeout handling**: Handle connection timeouts gracefully, with appropriate retry logic.

9. **Query optimization**: Monitor and optimize slow queries.

10. **Database schema migration**: Implement a versioned approach to database schema changes.

## Troubleshooting Common Issues

- **Connection Refused**: Ensure PostgreSQL is running and listening on the expected port.
- **Authentication Failed**: Verify the username and password.
- **Database Does Not Exist**: Ensure the 'tick_data' database has been created.
- **Table Does Not Exist**: Run the appropriate CREATE TABLE statements.
- **Slow Queries**: Check for missing indexes or overly complex queries.
- **Connection Pool Exhaustion**: Check for unclosed connections or increase the pool size.