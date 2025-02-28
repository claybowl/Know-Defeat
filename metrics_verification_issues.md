# Metrics Verification Issues

## Missing Metrics
1. **Win Streak Metrics Discrepancy**:
   - `metrics_calculator.py` calculates win_streak_2, win_streak_3, win_streak_4, win_streak_5
   - `bot_ranker.py` expects win_streak_2, win_streak_3, win_streak_4, win_streak_5, win_streak_6, win_streak_7
   - Need to add calculation for win_streak_6 and win_streak_7 in `metrics_calculator.py`

2. **Model Score Metrics Missing**:
   - `bot_ranker.py` expects: price_model_score, volume_model_score, price_wall_score
   - These don't appear to be calculated anywhere in `metrics_calculator.py`
   - Need to implement calculation methods for these metrics or adjust `bot_ranker.py` to handle their absence

## Data Storage Issues
1. **Algorithm ID Hardcoded**:
   - In `metrics_updater.py`, algo_id is hardcoded to 1 with a comment to replace with actual logic
   - Need to implement proper algo_id determination logic

2. **Database Constraint Issues**:
   - `metrics_updater.py` uses "ON CONFLICT (bot_id, algorithm_id)" suggesting both together form a unique key
   - `bot_ranker.py` uses "ON CONFLICT (bot_id)" when updating rankings, suggesting bot_id alone is unique
   - Need to verify database schema constraints match the assumptions in the code

3. **Update Implementation for Win Streaks**:
   - Win streaks are updated in a separate function call in `metrics_updater.py`
   - This could lead to race conditions or inconsistent data if there's an error between updates
   - Consider consolidating all updates into a single SQL statement

## Query Logic Issues
1. **Inconsistent Time Periods**:
   - Some performance metrics use NOW() in SQL queries, others pass datetime objects
   - Standardize the approach to time period calculations

2. **Default Values**:
   - Some functions use `result or 0.0` for NULL handling, others don't have explicit NULL handling
   - Implement consistent NULL handling across all metric calculations

## General Problems
1. **Error Handling**:
   - Most functions don't have specific error handling for database operations
   - Add more specific error handling and logging

2. **Metrics Validation**:
   - No validation that metric values are within reasonable ranges
   - Add validation to ensure calculated metrics make sense

## To-Do List
1. Update `metrics_calculator.py` to calculate all metrics expected by `bot_ranker.py`
2. Update `metrics_updater.py` to store all required metrics
3. Implement proper algo_id determination
4. Review and update database schema to ensure it matches code assumptions
5. Improve error handling and add validation
6. Add automated tests for metric calculations 