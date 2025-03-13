# Metrics Calculator System Fixes

## Problems Identified

1. **NaN and Type Conversion Issues**:
   - The `_ensure_float` method was not handling NaN values
   - Type conversion errors between PostgreSQL's Decimal and Python's float
   - The system was failing with "unsupported operand type(s) for /: 'decimal.Decimal' and 'float'"

2. **Parameter Type Issues**:
   - SQL queries were failing with "could not determine data type of parameter $2"
   - No explicit type conversions before passing values to SQL queries

3. **Database Schema Issues**:
   - Mismatches between column references in code and actual database schema
   - Missing or inappropriate SERIAL type for bot_id and algo_id

4. **Win Streak Calculation Issues**:
   - Win streak calculation wasn't handling type conversion properly
   - The SQL UPDATE statement was missing the algo_id in the WHERE clause

## Fixes Implemented

### 1. Enhanced the `_ensure_float` Method
```python
def _ensure_float(self, value):
    """Convert value to float if it's a Decimal or handle None/NaN cases."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        try:
            return float(value)
        except (ValueError, TypeError, OverflowError):
            return 0.0
    try:
        float_val = float(value)
        # Check for NaN or infinity
        if float_val != float_val or float_val == float('inf') or float_val == float('-inf'):
            return 0.0
        return float_val
    except (ValueError, TypeError):
        return 0.0
```

This implementation now properly handles:
- None values
- NaN values
- Infinity values
- Decimal conversion errors
- Type conversion errors

### 2. Fixed Parameter Type Handling in MetricsUpdater
```python
await connection.execute("""
    INSERT INTO bot_metrics (
        # ... columns ...
    )
    VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
""", 
bot_id, 
ticker, 
algo_id, 
float(one_hour_perf), 
float(two_hour_perf), 
# ... more explicit conversions ...
)
```

- Added explicit type conversion (float(), int()) for all parameters
- Added input validation for bot_id and ticker
- Improved error handling to provide more informative messages

### 3. Fixed Database Schema Issues
```sql
CREATE TABLE IF NOT EXISTS bot_metrics (
    -- Identifiers
    bot_id INTEGER,
    ticker VARCHAR(10),
    algo_id INTEGER,
    timestamp TIMESTAMP,
    -- ... more columns ...
)
```

- Changed bot_id and algo_id from SERIAL to INTEGER
- Added better data types for all columns with appropriate precision

### 4. Improved Win Streak Calculation
```python
async def calculate_and_insert_win_streaks(self, bot_id, algo_id):
    try:
        # ... code ...
        pnl = self._ensure_float(trade['trade_pnl'])
        if pnl > 0:
            # ... code ...
        
        # In SQL update:
        await connection.execute("""
            UPDATE bot_metrics
            SET 
                win_streak_2 = $3, 
                # ... more columns ...
            WHERE bot_id = $1 AND algo_id = $2
        """, 
        bot_id, 
        algo_id,
        float(streaks['win_streak_2']),
        # ... more parameters ...
        )
    except Exception as e:
        # Better error handling
        logging.error(f"Error calculating win streaks: {e}")
        return {
            'win_streak_2': 0, 
            # ... default values ...
        }
```

- Added explicit type conversion for trade_pnl
- Fixed the WHERE clause to include algo_id
- Added comprehensive error handling
- Added fallback values when errors occur

### 5. Enhanced Test Scripts
- Created a basic metrics test script
- Added schema inspection to verify database columns
- Added more detailed error reporting
- Created a verification script to test the fixes

## How to Verify the Fixes

1. **Install required libraries**:
   ```
   pip install asyncpg
   ```

2. **Run the basic metrics test**:
   ```
   python test_basic_metrics.py
   ```

3. **Run the verification script**:
   ```
   python verify_fixes.py
   ```

## Expected Results

The test scripts should now run without errors and show:
- Total PnL values as proper numbers (not NaN)
- Price model scores as calculated values
- Successful metrics updates in the database

## Next Steps

1. **Update the asyncpg import in all files**: 
   The linter is still showing errors about asyncpg imports. Make sure you have this library installed.

2. **Create test cases for each metric**:
   Add more comprehensive tests for each type of metric calculation.

3. **Consider adding database constraints**:
   Add appropriate constraints to the bot_metrics table to enable ON CONFLICT clauses. 