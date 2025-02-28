# Metrics System Fixes Summary

## Issues Identified

1. **Column Name Mismatches**:
   - The code was using `algorithm_id` but the actual database column is called `algo_id`
   - SQL queries were using `timestamp` but the correct column in `sim_bot_trades` is called `entry_time`

2. **Missing or Non-existent Columns**:
   - Model score calculations were relying on columns that don't exist:
     - `predicted_direction` (for price model score)
     - `volume_at_entry` and `avg_volume` (for volume model score)
     - `support_level` and `resistance_level` (for price wall score)

3. **SQL and Table Structure Issues**:
   - The `ON CONFLICT` clause in `metrics_updater.py` referenced a non-existent constraint
   - Methods in `MetricsCalculator` were querying a non-existent `trades` table
   - Some SQL queries lacked proper null handling, leading to division by zero errors

4. **Data Type Inconsistencies**:
   - Inconsistent handling of PostgreSQL's `decimal.Decimal` type vs. Python's `float`
   - Type errors when dividing: `unsupported operand type(s) for /: 'decimal.Decimal' and 'float'`

## Changes Made

### 1. Fixed Column Name References

- Updated all references from `algorithm_id` to `algo_id`:
  - In `MetricsUpdater`, `BotRanker`, and `AIWeightAdjuster` classes
  - Simplified to use `bot_id` directly as `algo_id` where appropriate

- Changed all `timestamp` references to `entry_time` in SQL queries:
  - Updated all time-based filtering (e.g., `WHERE entry_time >= NOW() - INTERVAL '1 hour'`)
  - Fixed sorting (`ORDER BY entry_time DESC`) and grouping (`GROUP BY DATE(entry_time)`)
  - Modified all date calculations to use the right column

### 2. Implemented Alternative Metrics Calculations

- Rewritten model score calculations to not rely on missing columns:
  - `calculate_price_model_score`: Now uses win rate and performance as proxies
  - `calculate_volume_model_score`: Uses trade frequency and profit per trade as indicators
  - `calculate_price_wall_score`: Uses profit factor and win rate to approximate effectiveness
  
- Added supporting methods:
  - Created `calculate_win_rate_over_period` to calculate win rates over specific timeframes
  - Enhanced existing methods with better error handling and fallbacks

### 3. Improved SQL Robustness

- Added null handling to prevent errors:
  - Used `NULLIF()` in division operations to prevent division by zero
  - Added fallback values for null results (e.g., `return result or 0.0`)
  
- Fixed the `ON CONFLICT` issue:
  - Removed the problematic constraint reference from the `metrics_updater.py` INSERT query
  - This allows basic inserting of metrics data while avoiding constraint errors

### 4. Enhanced Type Handling

- Added type conversion infrastructure:
  - Created `_ensure_float()` helper method to safely convert PostgreSQL's `Decimal` values to `float`
  - Applied this conversion to all methods that perform calculations
  
- Added explicit type handling:
  - Ensured consistent use of floating-point values in calculations
  - Added null checks with appropriate default values
  - Made type conversions explicit where needed

### 5. Additional Improvements

- Added comprehensive error handling:
  - Added try/except blocks with specific error messages
  - Provided reasonable fallback values (e.g., neutral scores of 50 for model metrics)
  
- Enhanced debugging support:
  - Created a basic test script (`test_basic_metrics.py`) to verify functionality
  - Added schema validation to identify column mismatches early

### Datetime Parameter Issues

- **Issue**: The `calculate_win_rate_over_period` function was called with `timedelta` objects instead of `datetime` objects, causing errors like: `invalid input for query argument $2: datetime.timedelta(days=1) (expected a datetime.date or datetime.datetime instance, got 'timedelta')`
- **Fix**:
  - Updated `calculate_win_rate_over_period` to accept an optional `period` parameter that can be a `timedelta` object
  - Modified the function to properly convert the `timedelta` to start and end datetime objects
  - Updated all calls to use the new parameter format (`period=timedelta(days=X)` instead of directly passing the timedelta)
  - Created a verification script to test the datetime parameter fixes

## Testing Strategy

The changes have been designed to make the metrics system more robust in the following ways:

1. **Schema Adaptability**: The code now adapts to the actual database schema rather than assuming column names
2. **Error Resilience**: Calculations now handle missing data, nulls, and type mismatches gracefully
3. **Fallback Mechanisms**: Where specific data is unavailable, reasonable proxies are used instead

### Database Datetime Testing

- Run the new `verify_datetime_fixes.py` script to test that datetime parameters are correctly handled in:
  - `calculate_win_rate_over_period`
  - `calculate_one_day_performance`
  - `calculate_one_week_performance`
  - `calculate_trade_frequency`
  - `calculate_price_model_score`
  - `calculate_price_wall_score`

## Next Steps

1. **Run Basic Tests**: 
   - Execute `test_basic_metrics.py` to verify the core functionality works
   - Check that all basic metrics can be calculated without errors

2. **Gradually Enable Advanced Features**:
   - Once basic metrics are confirmed working, test the full metrics pipeline
   - Enable the metrics updater to store results in the database

3. **Database Schema Alignment**:
   - Consider adding a `PRIMARY KEY` or `UNIQUE` constraint to the `bot_metrics` table to enable `ON CONFLICT` updates
   - Document the actual database schema to prevent future mismatches

4. **Long-term Improvements**:
   - Consider using an ORM (Object-Relational Mapping) to prevent column name issues
   - Implement a schema validation system to check for required columns at startup
   - Add proper migration scripts for database schema changes

The metrics system should now be more robust against database schema variations and missing data, while still providing valuable metrics for your trading bots. 

- Run the newly created verification script to test datetime parameter fixes: `python verify_datetime_fixes.py` 