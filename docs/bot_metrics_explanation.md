# Know-Defeat Bot Metrics System Documentation

This document explains the trading bot metrics calculations used in the Know-Defeat algorithmic trading system. It describes how each metric is calculated, what it represents, and how it affects bot ranking.

## Table of Contents

1. [Overview](#overview)
2. [Performance Period Metrics](#performance-period-metrics)
3. [Core Trading Metrics](#core-trading-metrics)
4. [Trade Statistics](#trade-statistics)
5. [Risk Metrics](#risk-metrics)
6. [Execution Metrics](#execution-metrics)
7. [Model Scores](#model-scores)
8. [Win Streak Analysis](#win-streak-analysis)
9. [Ranking System](#ranking-system)
10. [Fund Allocation Strategy](#fund-allocation-strategy)

## Overview

The Know-Defeat system tracks and calculates a wide range of metrics for each trading bot. These metrics are stored in the `bot_metrics` table and used to:

1. Evaluate bot performance
2. Rank bots against each other
3. Allocate trading funds
4. Make strategic improvements to algorithms

## Performance Period Metrics

These metrics measure bot performance over different time periods to evaluate both short-term and long-term effectiveness.

### One-Hour Performance
```sql
SELECT AVG(trade_pnl) AS performance
FROM sim_bot_trades
WHERE bot_id = $1 AND ticker = $2
AND entry_time >= NOW() - INTERVAL '1 hour';
```
- Represents: Average profit/loss per trade in the last hour
- Value Type: Dollar amount ($)
- Purpose: Evaluates very recent trading performance

### Two-Hour Performance
```sql
SELECT (SUM(CASE WHEN entry_time >= NOW() - INTERVAL '2 hours' THEN trade_pnl ELSE 0 END) /
NULLIF(SUM(CASE WHEN entry_time >= NOW() - INTERVAL '2 hours' THEN 1 ELSE 0 END), 0)) AS two_hour_performance
FROM sim_bot_trades
WHERE bot_id = $1 AND algo_id = $2;
```
- Represents: Average profit/loss per trade in the last 2 hours
- Value Type: Dollar amount ($)
- Purpose: Slightly longer view of recent performance

### One-Day Performance
```sql
-- Using a helper function that calculates:
SELECT SUM(trade_pnl) 
FROM sim_bot_trades
WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed' 
    AND entry_time BETWEEN [now - 1 day] AND [now];
```
- Represents: Total profit/loss in the last 24 hours
- Value Type: Dollar amount ($)
- Purpose: Evaluates daily trading performance

### One-Week Performance
```sql
-- Similar to one-day but with 1-week interval
```
- Represents: Total profit/loss in the last week
- Value Type: Dollar amount ($)
- Purpose: Evaluates medium-term performance consistency

### One-Month Performance
```sql
-- Similar to one-day but with 30-day interval
```
- Represents: Total profit/loss in the last 30 days
- Value Type: Dollar amount ($)
- Purpose: Evaluates longer-term performance stability

## Core Trading Metrics

These metrics form the foundation of bot evaluation, focusing on win rates and profits.

### Average Win Rate
```sql
SELECT 
    COUNT(*) as total_trades,
    SUM(CASE WHEN trade_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
FROM sim_bot_trades
WHERE bot_id = $1 AND ticker = $2

-- Then in code:
win_rate = (winning_trades / total_trades) * 100
```
- Represents: Percentage of trades that were profitable
- Value Type: Percentage (0-100%)
- Purpose: Measures consistency and prediction accuracy
- Note: Stored with precision DECIMAL(6,2) to allow for values up to 9999.99%

### Profit Per Second
```sql
SELECT 
    SUM(trade_pnl) as total_pnl,
    MAX(exit_time) - MIN(entry_time) as time_span
FROM sim_bot_trades
WHERE bot_id = $1

-- Then in code:
profit_per_second = total_pnl / time_span.total_seconds()
```
- Represents: Average profit generated per second of trading
- Value Type: Dollar amount per second ($/sec)
- Purpose: Measures trading efficiency and speed

### Total PnL
```sql
SELECT SUM(trade_pnl) AS total_pnl
FROM sim_bot_trades
WHERE bot_id = $1 AND algo_id = $2;
```
- Represents: Total profit/loss across all trades
- Value Type: Dollar amount ($)
- Purpose: Measures absolute performance

## Trade Statistics

These metrics provide deeper insights into trading behavior and effectiveness.

### Total Trades
```sql
SELECT COUNT(*) AS total_trades
FROM sim_bot_trades
WHERE bot_id = $1 AND algo_id = $2;
```
- Represents: Number of trades executed
- Value Type: Integer
- Purpose: Measures trading activity level

### Average Profit Per Trade
```sql
SELECT AVG(trade_pnl) AS avg_profit_per_trade
FROM sim_bot_trades
WHERE bot_id = $1 AND algo_id = $2;
```
- Represents: Average profit/loss per executed trade
- Value Type: Dollar amount ($)
- Purpose: Measures average effectiveness of individual trades

### Profit Factor
```sql
SELECT SUM(CASE WHEN trade_pnl > 0 THEN trade_pnl ELSE 0 END) /
    NULLIF(ABS(SUM(CASE WHEN trade_pnl < 0 THEN trade_pnl ELSE 0 END)), 0) AS profit_factor
FROM sim_bot_trades
WHERE bot_id = $1 AND algo_id = $2;
```
- Represents: Ratio of gross profits to gross losses
- Value Type: Ratio (higher is better)
- Purpose: Measures the efficiency of the trading strategy
- Note: Values above 1.0 indicate profitable systems; values below 1.0 indicate losing systems

## Risk Metrics

These metrics evaluate the risk characteristics of a trading bot.

### Average Drawdown
```sql
-- Complex calculation in code that:
-- 1. Gets all trades in order
-- 2. Calculates running PnL
-- 3. Tracks peaks and valleys in equity curve
-- 4. Records drawdowns (differences between peaks and subsequent valleys)
-- 5. Calculates average of all drawdowns
```
- Represents: Average decline from peak account value
- Value Type: Dollar amount ($)
- Purpose: Measures typical magnitude of losses during drawdowns

### Maximum Drawdown
```sql
-- Similar to Average Drawdown, but takes the maximum value
```
- Represents: Largest decline from peak account value
- Value Type: Dollar amount ($)
- Purpose: Measures worst-case historical loss scenario

### Sharpe Ratio
```sql
-- Complex calculation that:
-- 1. Groups trades by day to get daily returns
-- 2. Calculates average daily return
-- 3. Calculates standard deviation of daily returns
-- 4. Applies the Sharpe formula: (avg_return - risk_free_rate) / std_dev
```
- Represents: Risk-adjusted return measure
- Value Type: Ratio (higher is better)
- Purpose: Evaluates return relative to risk taken
- Note: Uses risk-free rate of 2% (0.02)

### Average True Range (ATR)
```sql
-- Complex calculation that:
-- 1. Gets daily high, low, and close prices
-- 2. Calculates true range for each day using maximum of:
--    a. High minus Low
--    b. |High minus Previous Close|
--    c. |Low minus Previous Close|
-- 3. Averages these values over the specified period (default 14 days)
```
- Represents: Average price volatility
- Value Type: Dollar amount ($)
- Purpose: Measures average price movement in the security
- Note: Used to adapt position sizes to market volatility

## Execution Metrics

These metrics evaluate how effectively trades are executed.

### Price Slippage
```sql
-- Calculated in the calculate_and_insert_execution_metrics method as:
price_slippages = [abs(trade['exit_price'] - trade['entry_price']) for trade in trades]
price_slippage = average of price_slippages
```
- Represents: Average price difference between entry and exit
- Value Type: Dollar amount ($)
- Purpose: Measures execution quality and price differences

### Average Trade Duration
```sql
-- Calculated in the calculate_and_insert_execution_metrics method as:
time_slippages = [(trade['exit_time'] - trade['entry_time']).total_seconds() for trade in trades]
avg_trade_duration = average of time_slippages (converted to timedelta)
```
- Represents: Average time trades are held
- Value Type: Time duration (e.g., "00:15:30" for 15 minutes, 30 seconds)
- Purpose: Measures typical trade holding period

## Model Scores

These metrics evaluate the effectiveness of various prediction models used by the bots.

### Price Model Score
```sql
-- Calculated based on win rate and recent performance:
one_day_perf = await self.calculate_one_day_performance(bot_id, algo_id)
win_rate = await self.calculate_win_rate_over_period(bot_id, algo_id, period=timedelta(days=1))
score = (win_rate * 0.7) + (min(max(one_day_perf, 0), 100) * 0.3)
```
- Represents: Effectiveness of price prediction components
- Value Type: Score (0-100)
- Purpose: Evaluates price prediction accuracy

### Volume Model Score
```sql
-- Calculated based on trading frequency and profit per trade:
daily_trade_count = await self.calculate_trade_frequency(bot_id, algo_id, timedelta(days=1))
profit_per_trade = await self.calculate_avg_profit_per_trade(bot_id, algo_id)
score = min(daily_trade_count * profit_per_trade * 5, 100) if conditions met else 50
```
- Represents: Effectiveness of volume analysis components
- Value Type: Score (0-100)
- Purpose: Evaluates volume-based prediction accuracy

### Price Wall Score
```sql
-- Calculated based on profit factor and win rate:
profit_factor = await self.calculate_profit_factor(bot_id, algo_id) or 1
win_rate = await self.calculate_win_rate_over_period(bot_id, algo_id, period=timedelta(days=7))
score = min((profit_factor * 10) + (win_rate * 0.5), 100) if profit_factor > 0 else 50
```
- Represents: Effectiveness of support/resistance detection
- Value Type: Score (0-100)
- Purpose: Evaluates price barrier identification accuracy

## Win Streak Analysis

These metrics analyze consecutive winning trades to measure consistency and momentum.

### Win Streak Metrics (2, 3, 4, 5 consecutive wins)
```sql
-- Complex logic in calculate_and_insert_win_streaks that:
-- 1. Gets all trades in chronological order
-- 2. Tracks consecutive winning trades
-- 3. Counts occurrences of streaks of length 2, 3, 4, and 5
-- 4. Calculates percentage of total trades these streaks represent
```
- Represents: Frequency of consecutive winning trades
- Value Type: Percentage (0-100%)
- Purpose: Measures trading consistency and momentum
- Note: Higher streaks (4, 5) are weighted more in the ranking system

## Ranking System

Bots are ranked using a weighted score of all the above metrics. The weights reflect the relative importance of each metric for overall performance.

### Weight System
From the `get_variable_weights` method in `BotRanker`:

```python
weights = {
    # Performance periods - more recent periods have higher weight
    'one_hour_performance': 15.0,   # Most recent performance is highly valued
    'two_hour_performance': 10.0,
    'one_day_performance': 12.0,
    'one_week_performance': 8.0,
    'one_month_performance': 5.0,   # Long-term stability is still important
    
    # Core metrics - fundamental indicators of success
    'avg_win_rate': 12.0,          # Consistency in winning trades
    'profit_per_second': 10.0,     # Efficiency of profit generation
    'total_pnl': 8.0,              # Total profit is important
    
    # Trade statistics
    'profit_factor': 8.0,          # Ratio of gains to losses
    'avg_profit_per_trade': 6.0,   # Average profit per trade
    
    # Risk metrics - lower is better
    'avg_drawdown': -5.0,          # Negative weight as lower drawdown is better
    'max_drawdown': -7.0,          # Negative weight as lower max drawdown is better
    'sharpe_ratio': 8.0,           # Risk-adjusted return measure
    
    # Model scores - algorithmic effectiveness
    'price_model_score': 9.0,      # Effectiveness of price prediction
    'volume_model_score': 7.0,     # Effectiveness of volume analysis
    'price_wall_score': 6.0,       # Effectiveness of support/resistance analysis
    
    # Win streaks - psychological and momentum indicators
    'win_streak_2': 3.0,
    'win_streak_3': 4.0,
    'win_streak_4': 5.0,
    'win_streak_5': 6.0,
}
```

### Rank Score Calculation
```python
# For each metric:
score += weight * value

# For metrics where lower is better (negative weights):
weight = abs(weight)
adjusted_value = 100.0 - min(value, 100.0)  # Invert the scale
score += weight * adjusted_value

# Normalization to keep scores in reasonable range
score = score / 100.0
```

## Fund Allocation Strategy

The system uses a fixed allocation strategy where each trade gets exactly 10% of the total funds (by default, $2,000 out of $20,000). This allows for a maximum of 10 concurrent trades at any time.

### Allocation Logic
```python
# Fixed 10% allocation per trade
fixed_trade_amount = total_funds / 10  # $2,000 with default $20,000 total

# For each active trade:
allocations.append({
    'bot_id': bot_id,
    'ticker': ticker,
    'rank_score': rank_score,
    'rank': rank,
    'trade_id': trade_id,
    'allocation_amount': fixed_trade_amount,
    'allocation_percentage': 10.0
})

# Total allocation percentage
allocated_pct = len(allocations) * 10.0
```

### Selection Priority
Higher-ranked bots get priority for trading slots when multiple bots are competing for limited positions. This ensures that the most effective strategies get precedence in fund allocation.