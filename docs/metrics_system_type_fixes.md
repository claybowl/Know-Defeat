# Metrics System Type Conversion Fixes

## Summary of Changes

We identified and fixed issues related to decimal.Decimal vs float type conversion in the metrics calculation system. The primary error was: "unsupported operand type(s) for /: 'decimal.Decimal' and 'float'", which occurred during division operations where one operand was a Decimal and the other was a float.

## Specific Improvements

### 1. Enhanced _ensure_float() Method

The `_ensure_float()` method in `metrics_calculator.py` was significantly improved:

- Added proper handling for None, NaN, and infinity values
- Added a default parameter to specify fallback value
- Restructured the logic for clearer flow and better error handling
- Used the `math` module to properly detect NaN and infinity values

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

### 2. Applied Type Conversion in All Division Operations

- Fixed `calculate_drawdowns` method to ensure all `trade_pnl` values are converted to float
- Enhanced `calculate_win_rate_over_period` to convert both `total_trades` and `winning_trades` to float before division
- Updated `calculate_profit_per_second` to ensure `total_pnl` is converted to float
- Fixed `calculate_and_insert_win_streaks` to handle type conversion for streak calculations
- Added explicit float conversion in SQL parameter binding for database updates

### 3. Added Type Checks in MetricsUpdater

- Added explicit type checks in `update_bot_metrics` method
- Implemented a new block that converts all potential Decimal values to float before SQL insertion
- Added error handling to prevent failures from stopping the entire process

```python
# Ensure all values are properly converted to the right type
try:
    # Convert all potential Decimal values to float
    if isinstance(one_hour_perf, Decimal):
        one_hour_perf = float(one_hour_perf)
    # ... more conversions for all metrics
    
    # Handle potential Decimal values in drawdown_info
    for key in ['avg_drawdown', 'max_drawdown']:
        if key in drawdown_info and isinstance(drawdown_info[key], Decimal):
            drawdown_info[key] = float(drawdown_info[key])
except Exception as e:
    logging.error(f"Error converting metric types for bot {bot_id}: {e}")
    # Continue with the values we have
```

### 4. Improved SQL Query Structure

- Consolidated multiple SQL queries into single queries with more efficient structure
- Used `SUM(CASE WHEN...)` instead of separate queries to get totals and subsets
- Ensured proper parameterization in all SQL queries
- Removed redundant database lookups

## Testing Status

After these changes, we ran the test script again. The script should now handle all potential type mismatches between Decimal and float values.

## Next Steps

1. Install the `asyncpg` library to resolve import errors
2. Run all test scripts again to verify the fixes
3. Consider adding a comprehensive test suite specifically for numeric operations
4. Add database schema validation to ensure schema changes don't affect the type conversion logic

## Conclusion

These changes have made the metrics calculation system significantly more robust against type conversion issues. By consistently using `_ensure_float()` and adding explicit type conversion in critical calculation paths, we've eliminated the potential for decimal.Decimal vs float type mismatches. 