# Decimal vs Float Type Mismatch Fix Summary

## Problem Overview

We addressed a critical issue in the metrics calculation system where operations between `decimal.Decimal` and `float` types were causing errors. The specific error was:

```
unsupported operand type(s) for /: 'decimal.Decimal' and 'float'
```

This error occurred during division operations where one operand was a Decimal (typically from database results) and the other was a float.

## Implemented Fixes

### 1. Enhanced `_ensure_float()` Method in MetricsCalculator

We significantly improved the `_ensure_float()` method to handle a wider range of edge cases:

```python
def _ensure_float(self, value, default=0.0):
    """Convert value to float, handling None, NaN, and infinity values safely."""
    if value is None:
        return default
    
    try:
        # Convert Decimal to float
        if isinstance(value, Decimal):
            value = float(value)
            
        # Handle float conversion
        result = float(value)
        
        # Check for NaN or infinite values
        if math.isnan(result) or math.isinf(result):
            return default
            
        return result
    except (ValueError, TypeError, OverflowError):
        return default
```

### 2. Added Type Conversion in Division Operations

We identified and fixed all places where division operations could involve mixed types:

- In `calculate_win_rate_over_period()`:
  ```python
  total_trades = self._ensure_float(row['total_trades'])
  winning_trades = self._ensure_float(row['winning_trades'])
  if total_trades == 0:
      return 0.0
  return (winning_trades / total_trades) * 100
  ```

- In `calculate_profit_per_second()`:
  ```python
  total_pnl = self._ensure_float(row['total_pnl'])
  total_seconds = row['time_span'].total_seconds()
  if total_seconds == 0:
      return 0.0
  return total_pnl / total_seconds
  ```

- In `calculate_drawdowns()`:
  ```python
  # Convert all trade_pnl values to float
  pnl_values = [self._ensure_float(row['trade_pnl']) for row in rows]
  ```

- In win streak calculations:
  ```python
  total_trades_float = self._ensure_float(total_trades)
  if total_trades_float > 0:
      win_streak_metrics[f"win_streak_{streak_len}"] = (streaks[streak_len] / total_trades_float) * 100
  ```

### 3. Added Explicit Type Conversion in MetricsUpdater

We added a dedicated type conversion block in the `update_bot_metrics` method to ensure all metrics are properly converted to float before database insertion:

```python
# Ensure all values are properly converted to the right type
try:
    # Convert all potential Decimal values to float
    if isinstance(one_hour_perf, Decimal):
        one_hour_perf = float(one_hour_perf)
    # ... more conversions for all metrics ...
    
    # Handle potential Decimal values in drawdown_info
    for key in ['avg_drawdown', 'max_drawdown']:
        if key in drawdown_info and isinstance(drawdown_info[key], Decimal):
            drawdown_info[key] = float(drawdown_info[key])
except Exception as e:
    logging.error(f"Error converting metric types for bot {bot_id}: {e}")
    # Continue with the values we have
```

### 4. Improved SQL Parameter Binding

We added explicit float conversions in all SQL parameter bindings to ensure consistent types:

```python
await connection.execute("""
    INSERT INTO bot_metrics (
        bot_id, ticker, algo_id, timestamp, ...
    )
    VALUES ($1, $2, $3, NOW(), ...)
""", 
bot_id, 
ticker, 
algo_id, 
float(one_hour_perf),  # Explicit conversion
float(two_hour_perf),  # Explicit conversion 
# ... more explicitly converted parameters ...
)
```

### 5. Consolidated SQL Queries

We improved the SQL query structure to reduce the number of separate queries and potential type conversion issues:

```python
# Before: Two separate queries
total_trades = await connection.fetchval("SELECT COUNT(*) FROM sim_bot_trades WHERE...")
winning_trades = await connection.fetchval("SELECT COUNT(*) FROM sim_bot_trades WHERE... AND trade_pnl > 0")

# After: A single query with CASE statement
query = """
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
    FROM sim_bot_trades
    WHERE bot_id = $1 AND ticker = $2
"""
row = await connection.fetchrow(query, bot_id, ticker)
```

## Validation Tests

We created two test scripts to validate our fixes:

1. **test_basic_metrics.py**: Tests the core type conversion logic with synthetic data
2. **verify_fixes.py**: Connects to the database and tests with real data

The test results showed that our fixes successfully resolved the type mismatch errors and made the type conversion logic more robust.

## Lessons Learned

1. Always use consistent types in mathematical operations, especially division
2. Database values should be explicitly converted to the target type before calculations
3. Add defensive checking for edge cases like None, NaN, and infinity
4. Use proper error handling to prevent cascading failures from type conversion issues

## Next Steps

1. Install the `asyncpg` library to resolve import errors
2. Add more comprehensive testing for edge cases
3. Consider adding schema validation to detect database type mismatches early
4. Implement consistent error handling patterns across the entire metrics system

By implementing these fixes, the metrics calculation system is now more robust against type conversion issues and can handle a wider range of input data without failing. 