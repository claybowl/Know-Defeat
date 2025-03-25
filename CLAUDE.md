# Know-Defeat Algorithmic Trading System Guide

# Instructions and Information for Claude

- Check your knowledge-graph-memory when responding to user prompts.
- After acheiving success in a task, or series of task, add that information to the knowledge-graph-memory.
- When developing or learning new features of the Know Defeat system, add that information to the knowlege-graph-memory.

- I'm using anaconda environment for this project. I'm also using bash terminal. The conda environment i'm using is:
  - `conda activate Autogen`

- I'm using PostgreSQL database. Here's how to start the database: 
  - `pg_ctl -D "C:/Users/clayb/postgres_data" start`
  - `psql -U clayb -d tick_data`

- When creating new files that generate logs, please save those logs in my logs/ directory. When generating documents, please save them in my docs/ directiory.

# Interactive Brokers Integration

## IB Gateway Setup
- IB Gateway must be running on port 4002
- Client ID 0 is reserved for the main data ingestion
- Each bot gets assigned a unique client ID matching its bot_id

## IB Controller Configuration
```python
# Connection parameters
host = '127.0.0.1'
port = 4002
client_id = 0  # Main controller uses ID 0

# Connection initialization
app.connect(host, port, client_id)
```

## Market Data Subscription Tiers
```python
# Priority tier - most liquid/important symbols
TIER_1_SYMBOLS = [
    'TSLA',  # Tesla
    'COIN',  # Coinbase
    'SPY',   # S&P 500 ETF
    'QQQ',   # Nasdaq ETF
    'AAPL'   # Apple
]

# Secondary tier symbols
TIER_2_SYMBOLS = [
    'MSFT',  # Microsoft
    'NVDA',  # NVIDIA
    'META'   # Meta
]
```

## Common IB API Issues
- Connection refused: Check if IB Gateway is running
- Market data subscription failed: Verify market data permissions
- Pacing violation: Respect IB API rate limits
- Reconnection handling: System auto-reconnects after disconnection

# Bot Configuration Format

## YAML Configuration Example
```yaml
bot_id: 1
name: "TSLA_Breakout_Bot"
ticker: "TSLA"
algorithm_module: "algorithms.breakout_algorithm"
algorithm_type: "breakout"
trade_direction: "BOTH"
position_size: 1000.0
trailing_stop_pct: 0.01
description: "TSLA breakout strategy using volatility-based entry"
version: "1.0"
parameters:
  lookback_period: 20
  volatility_threshold: 2.0
  profit_target_pct: 0.02
```

## Bot Types and Algorithms
- Breakout strategies
- Mean reversion strategies
- Price pattern recognition
- Support/resistance based
- Volatility breakout systems

## Bot Ranking System
The system ranks bots based on:
- Win rate
- Profit factor
- Risk-adjusted returns
- Maximum drawdown
- Sharpe ratio
- Recent performance

# Current System Status

## Implemented Features
- Real-time market data ingestion from IB
- Bot registration and management
- Trade execution and monitoring
- Performance metrics calculation
- Basic bot ranking system
- Database logging and analysis

## In Development
- Enhanced multi-agent coordination
- Advanced risk management protocols
- Machine learning-based strategy optimization
- Real-time performance analytics
- Automated strategy adjustment

## Planned Features
- Multi-agent collaboration system (AIGentic)
- High-throughput message bus
- Advanced probability engine
- Dynamic weight adjustment system
- Curve AI autonomous trading system

# PostgreSQL Database Schema for Know-Defeat Trading System

This section provides comprehensive information about the database tables, relationships, and design patterns used in the Know-Defeat algorithmic trading system.

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
    win_rate NUMERIC(5,4) NOT NULL DEFAULT 0,
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

## Build/Test Commands

- Run all tests: `python -m pytest tests/`
- Run specific test: `python -m pytest tests/test_bot_ranker.py::test_fetch_bot_metrics`
- Run direct test file: `python tests/test_basic_metrics.py`
- Type checking: `mypy src/ algorithms/`
- Lint code: `flake8 src/ algorithms/ tests/`
- Run Streamlit UI: `streamlit run user_interface/src/streamlit_app2.py`
- Export trades: `python user_interface/src/export_all_trades.py`
- Register bots: `bash register_bots.sh` or `python scripts/register_all_bots.py`

- 1. test_trade_creation.py - A simple script that creates a single test trade and updates metrics for one bot 
  2. test_metrics_system.py - A more comprehensive script that checks all bots, their trade counts, and updates metrics for bots with trades
  3. test_trading_pipeline.py - A full end-to-end test that simulates the trading pipeline from trade creation through metrics calculation and bot ranking

  To run these tests and ensure the system is working correctly, you can execute them in the following order:  

  ### First, run the simple trade creation test
  python tests/test_trade_creation.py

  ### Then check the overall metrics system
  python tests/test_metrics_system.py

  ### Finally, run the full pipeline test
  python tests/test_trading_pipeline.py

  These tests will:
  1. Create test trades for various bots
  2. Update metrics based on those trades
  3. Verify metrics are calculated correctly
  4. Rank bots based on their metrics
  5. Calculate fund allocations based on rankings

  By running these tests, you can confirm that your system is correctly:
  - Recording trades in the sim_bot_trades table
  - Calculating metrics based on those trades
  - Storing metrics in the bot_metrics table
  - Ranking bots based on their metrics
  - Allocating funds based on rankings

## Code Style Guidelines

- **Imports**: Standard library first, then third-party, then local modules
- **Types**: Use type hints (Dict, List, Any, Optional) for arguments and returns
- **Docstrings**: Google-style with Args/Returns sections
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error handling**: Use specific exception types with contextual logging
- **Logging**: Appropriate levels (debug, info, warning, error) with context
- **Async**: Use asyncio patterns with proper async/await and context managers
- **Database**: Use asyncpg for async database operations
- **Indentation**: 4 spaces, consistent throughout codebase

## Project Structure

- `algorithms/`: Trading algorithm implementations
- `src/`: Core system components and utilities
- `tests/`: Test suite using pytest
- `user_interface/`: Streamlit-based UI components
- `scripts/`: Utility scripts for database operations and bot management

# Project Vision and Roadmap

The Know Defeat project aims to create a sophisticated algorithmic trading system that combines high-frequency trading capabilities with autonomous agent intelligence. Our key milestones include:

## Current Phase
- Real-time market data processing with IB API integration
- Basic bot management and execution system
- Performance tracking and analysis
- Initial implementation of TSLA and COIN strategies

## Next Phase (Q2 2024)
- Enhanced multi-agent coordination system
- Advanced risk management protocols
- Machine learning strategy optimization
- Real-time performance analytics dashboard

## Future Development (Q3-Q4 2024)
- AIGentic System for collaborative decision-making
- High-throughput message bus implementation
- Advanced probability engine
- Dynamic weight adjustment system
- Curve AI autonomous trading system

Our vision is to create a trading ecosystem where autonomous agents collaborate and compete to discover and exploit market opportunities while maintaining robust risk management. The system will continuously learn and adapt to changing market conditions through collective intelligence.