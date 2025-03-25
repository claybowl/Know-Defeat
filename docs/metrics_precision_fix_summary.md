# Database Numeric Precision Fix

## Issue

We identified an issue with numeric field overflow errors when updating metrics for certain bots. Specifically, bots 3 (TSLA) and 5 (NVDA) were failing with the error:

> numeric field overflow. Detail: A field with precision 6, scale 2 must round to an absolute value less than 10^4.

This means that some metric values were exceeding the allowed range for `DECIMAL(6,2)` columns, which can only store values between -9999.99 and 9999.99.

## Root Cause

The `bot_metrics` table has several columns defined with `DECIMAL(6,2)` precision, including:

- one_hour_performance
- two_hour_performance
- one_day_performance
- avg_win_rate
- avg_drawdown
- max_drawdown
- price_model_score
- volume_model_score
- price_wall_score
- win_streak_2 through win_streak_5

When calculations for these metrics produced values outside the range of -9999.99 to 9999.99, the database would reject the values with a numeric overflow error. This was especially happening for high-performing bots like TSLA and NVDA.

## Fix Implementation

We implemented a solution to limit all metric values to stay within their respective database column constraints:

1. Added a `_limit_decimal_value` helper function to both `MetricsCalculator` and `MetricsUpdater` classes that:
   - Takes a value along with precision and scale parameters
   - Calculates the maximum allowed value based on precision and scale
   - Limits the value to stay within the allowed range
   - Rounds the value to the specified scale

2. Modified all metric calculation methods to apply these limits before returning values
   - All `DECIMAL(6,2)` fields are now capped at +/-9999.99
   - All other numeric fields are capped at their respective precision limits
   
3. Updated the `update_bot_metrics` method to apply limits to all values before inserting them into the database

4. Added special handling for win streak calculations to ensure they also respect the precision constraints

## Benefits

- No more numeric overflow errors when updating metrics
- Consistent handling of large numeric values across the system
- Values are capped at reasonable limits rather than failing
- Performance metrics continue to work even for very high-performing bots

## Future Considerations

For the future, if we need to store larger values, we could:

1. Modify the database schema to increase the precision of certain columns
2. Add scaling factors to store large values in a normalized form
3. Implement logarithmic scaling for metrics that can grow very large

For now, the current solution provides a balance between accuracy and database compatibility without requiring schema changes.