# Metrics Column Fix and Schema Update Summary

## Issues Identified
1. The code was using `algorithm_id` but the actual database column is called `algo_id`
2. Complex logic to determine `algo_id` was causing additional issues
3. The `total_trades` column existed in the database but wasn't being populated
4. Methods in the `MetricsCalculator` were querying a non-existent `trades` table
5. SQL queries were using a "timestamp" column but the actual column in `sim_bot_trades` is called `entry_time`
6. The model score calculation methods were looking for columns that don't exist in the database
7. The ON CONFLICT clause in the metrics_updater was referencing a constraint that doesn't exist
8. There were type mismatches between decimal.Decimal and float in calculations

## Changes Made

### 1. Updated Database Schema
- Implemented a new optimized schema for the `bot_metrics` table with:
  - Improved column definitions with appropriate data types and precision
  - Better organization of metrics by category (identifiers, performance, risk metrics, etc.)
  - Added additional metrics like `avg_profit_per_trade`, `trade_frequency`, and `max_drawdown`

### 2. Updated MetricsUpdater Class
- Removed the `get_algo_id` method completely
- Simplified logic to always use `bot_id` as `algo_id` directly
- Updated CREATE TABLE statement to use the new optimized schema
- Added code to calculate and insert additional metrics (`total_pnl`, `total_trades`, `avg_profit_per_trade`)
- Updated SQL statements to match the new schema
- Removed the ON CONFLICT clause that was causing errors due to missing constraints

### 3. Updated MetricsCalculator Class
- Fixed methods that were querying the non-existent `trades` table:
  - Changed SQL queries to use the `sim_bot_trades` table instead
  - Revised `calculate_avg_win_rate` to compute the win rate from actual trades
  - Modified `calculate_one_hour_performance` to use `trade_pnl` from `sim_bot_trades`
- Updated `calculate_and_insert_win_streaks` method:
  - Modified to use the new table structure
  - Updated to use `entry_time` instead of `timestamp` for all SQL queries
  - Reduced to tracking only 4 win streak metrics (2-5) instead of 6 (2-7)
- Fixed timestamp column references:
  - Replaced all instances of `timestamp` with `entry_time` in SQL queries
  - Updated all datetime filtering and ordering to use the correct column
- Fixed model score calculations:
  - Reimplemented `calculate_price_model_score`, `calculate_volume_model_score`, and `calculate_price_wall_score` 
    to not rely on missing columns like `predicted_direction`, `volume_at_entry`, and `support_level`
  - Created a new `calculate_win_rate_over_period` helper method
  - Used trade frequency, profit per trade, and win rate as proxies for model scores
- Added data type conversions:
  - Created `_ensure_float` helper method to convert decimal.Decimal to float
  - Added null checks and default values to prevent errors
  - Added NULLIF in SQL queries to prevent division by zero
  - Applied type conversions to all methods that perform calculations

### 4. Updated Test Script
- Enhanced the `test_metrics_calculation.py` script to:
  - Inspect the schema of `sim_bot_trades` to identify available columns
  - Check for critical columns like `bot_id`, `algo_id`, and `trade_pnl`
  - Identify which timestamp-related column is available (`timestamp`, `trade_time`, etc.)
  - Update expected metrics list to match new schema

## Timestamp Column Issue Resolution
- Identified that the `sim_bot_trades` table uses `entry_time` instead of `timestamp`
- Updated all SQL queries that were referencing `timestamp` to use `entry_time` instead
- Fixed all time-based calculations to use the correct column, including:
  - Performance calculations (hourly, daily, weekly, monthly)
  - Win rate calculations
  - Drawdown analysis
  - Date-based grouping operations

## Missing Columns Solution
- Created alternative implementations for metrics that relied on columns not present in the database:
  - For price model score: Used win rate and daily performance as proxies
  - For volume model score: Used trade frequency and profit per trade as indicators
  - For price wall score: Used profit factor and win rate as approximations
- Added error handling with reasonable fallback values (neutral scores)
- Ensured calculation methods are robust to missing or null data

## Type Conversion Fixes
- Added a helper method `_ensure_float` to convert decimal.Decimal values to float
- Applied this conversion to all methods that perform calculations
- Added NULLIF to SQL queries to prevent division by zero errors
- Added null checks with default values (0.0, 1.0, etc.) to prevent NoneType errors
- Used explicit type conversion to float to ensure consistency in calculations

## Next Steps

1. **Run the Test Script Again**
   - Execute `python test_metrics_calculation.py` with proper database credentials
   - The script should now work correctly with the updated column references
   - All SQL queries now reference `entry_time` instead of `timestamp`

2. **Test Individual Components**
   - Test the `metrics_calculator.py` methods individually to ensure they correctly calculate metrics
   - Test the `metrics_updater.py` methods to ensure they correctly store metrics

3. **Verify Bot Metrics Table Creation**
   - Check that the `bot_metrics` table is properly created with the new optimized schema
   - Run SQL queries directly on the database to inspect the table structure

4. **Final Integration Testing**
   - Ensure the complete metrics pipeline works properly
   - Verify that metrics are correctly calculated and stored for multiple bots

## Future Improvements
- Add more validation when accessing database columns to catch mismatches early
- Create a database schema migration script to ensure consistency between code and database
- Add unit tests for database operations to catch schema issues
- Consider using an ORM (Object-Relational Mapping) to avoid SQL name issues
- Implement a data dictionary or central repository of column definitions 