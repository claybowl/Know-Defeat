# Bot Metrics Validation Guide

This guide documents the key metrics used in the KnowDefeat trading system, including their formulas, expected ranges, interpretation, and potential improvements.

## Core Performance Metrics

### Time-Period Based Performance

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `one_hour_performance` | `SUM(trade_pnl)` for trades in last hour | Any numeric value | Short-term performance indicator |
| `two_hour_performance` | `SUM(trade_pnl)` for trades in last 2 hours | Any numeric value | Short-term performance indicator |
| `one_day_performance` | `SUM(trade_pnl)` for trades in last day | Any numeric value | Daily performance indicator |
| `one_week_performance` | `SUM(trade_pnl)` for trades in last week | Any numeric value | Medium-term performance indicator |
| `one_month_performance` | `SUM(trade_pnl)` for trades in last 30 days | Any numeric value | Long-term performance indicator |

**Potential Improvements:**
- Normalize by trading volume to get percentage returns
- Add performance variance or standard deviation
- Add relative performance compared to market benchmark
- Adjust for risk by dividing by maximum drawdown during period

### Win Rate Metrics

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `avg_win_rate` | `(COUNT(winning_trades) / COUNT(total_trades)) * 100` | 0 to 100 | Percentage of profitable trades |
| `win_streak_2` | Percentage of trades that are part of at least 2 consecutive wins | 0 to 100 | Consistency of short winning streaks |
| `win_streak_3` | Percentage of trades that are part of at least 3 consecutive wins | 0 to 100 | Consistency of medium winning streaks |
| `win_streak_4` | Percentage of trades that are part of at least 4 consecutive wins | 0 to 100 | Consistency of medium-long winning streaks |
| `win_streak_5` | Percentage of trades that are part of at least 5 consecutive wins | 0 to 100 | Consistency of long winning streaks |

**Potential Improvements:**
- Separate win rates by trade direction (long vs short)
- Add loss streaks as complementary metrics
- Add time-weighted win rate (more recent trades weighted higher)
- Add expectancy calculation (avg_win * win_rate - avg_loss * loss_rate)

### Profitability Metrics

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `total_pnl` | `SUM(trade_pnl)` for all trades | Any numeric value | Total profit/loss |
| `avg_profit_per_trade` | `AVG(trade_pnl)` for all trades | Any numeric value | Average profit per trade |
| `profit_per_second` | `total_pnl / (MAX(exit_time) - MIN(entry_time))` | Any numeric value | Trading efficiency over time |
| `profit_factor` | `SUM(winning_trade_pnl) / ABS(SUM(losing_trade_pnl))` | ≥0 (>1 is profitable) | Ratio of gross profits to gross losses |

**Potential Improvements:**
- Add R-multiple (avg_win / avg_loss ratio)
- Add expectancy score (win_rate * avg_win - loss_rate * avg_loss)
- Add adjusted profit factor that accounts for trading frequency
- Add net profit factor that includes fees and slippage

## Risk Metrics

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `avg_drawdown` | Average of all drawdowns in equity curve | ≥0 | Average depth of temporary losses |
| `max_drawdown` | Maximum peak-to-trough decline in equity | ≥0 | Worst-case historical loss |
| `time_in_drawdown` | Total time spent in drawdown state | Time interval | Recovery speed indicator |
| `sharpe_ratio` | `(avg_daily_return - risk_free_rate) / std_dev_daily_return` | Typically -3 to +4 | Risk-adjusted return (>1 good, >2 very good) |
| `average_true_range` | Average daily price ranges over period | >0 | Market volatility indicator |

**Potential Improvements:**
- Add Sortino ratio (focuses on downside risk only)
- Add Calmar ratio (annual return / maximum drawdown)
- Add maximum drawdown duration
- Add drawdown recovery metrics
- Add Value at Risk (VaR) metrics
- Add Maximum Adverse Excursion (MAE)

## Execution Metrics

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `price_slippage` | `AVG(ABS(exit_price - entry_price))` | ≥0 | Average price change during trade |
| `time_slippage` | Average time between trigger and execution | Time interval | Execution speed |
| `avg_trade_duration` | `AVG(exit_time - entry_time)` | Time interval | Average holding period |

**Potential Improvements:**
- Separate metrics for entry vs exit slippage
- Add metrics for execution quality relative to market conditions
- Add percentage-based slippage metrics (slippage/price)
- Add comparison of actual vs optimal execution timing

## Model Score Metrics

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `price_model_score` | Composite score based on price prediction accuracy | 0 to 100 | Price prediction model effectiveness |
| `volume_model_score` | Composite score based on volume prediction accuracy | 0 to 100 | Volume prediction model effectiveness |
| `price_wall_score` | Score based on support/resistance level prediction | 0 to 100 | Order book prediction effectiveness |

**Potential Improvements:**
- Replace simplified implementations with actual model prediction vs outcome comparisons
- Add directional accuracy metrics (correct price movement prediction percentage)
- Add magnitude accuracy metrics (price level prediction accuracy)
- Separate scores for bullish vs bearish predictions

## Activity Metrics

| Metric | Formula | Expected Range | Interpretation |
|--------|---------|----------------|----------------|
| `total_trades` | `COUNT(trades)` | ≥0 | Total number of completed trades |
| `trade_frequency` | Number of trades per time unit | ≥0 | Trading activity level |

**Potential Improvements:**
- Add metrics for optimal trade frequency
- Add metrics for intraday trade distribution
- Add comparison to market opportunity count

## Calculation Issues & Troubleshooting

When metrics show unexpected values, consider:

1. **NULL values**:
   - Check if trades have NULL values in critical fields like entry_time, exit_time, or trade_pnl
   - Verify proper NULL handling in calculation methods

2. **Decimal precision issues**:
   - Check for potential overflow or underflow in calculations
   - Verify proper rounding to match database column precision

3. **Time period boundaries**:
   - For time-based metrics, verify correct timezone handling
   - Check date logic for edge cases like month transitions

4. **Division by zero**:
   - Verify that denominators in ratio calculations can't be zero
   - Check NULLIF usage in SQL queries

5. **Data consistency**:
   - Ensure all trades have proper status values
   - Check for duplicate trades affecting calculations

## New Metrics to Consider

1. **Calmar Ratio**: Annual return / Maximum drawdown
   - Measures return relative to risk
   - Higher values indicate better risk-adjusted performance

2. **Sortino Ratio**: (Return - Risk-free rate) / Downside deviation
   - Similar to Sharpe but only penalizes harmful volatility

3. **Maximum Consecutive Losses**: Longest streak of losing trades
   - Important risk indicator to complement win streaks

4. **Return on Maximum Drawdown (RoMaD)**: Total return / Maximum drawdown
   - Measures how efficiently the system recovers from drawdowns

5. **Optimal Trade Size**: Calculated based on Kelly criterion
   - Optimizes position sizing based on win rate and profit factor

6. **Trade Efficiency**: Actual profit / Theoretical maximum profit
   - Measures how optimally trades are entered and exited

7. **Directional Bias Score**: Success rate comparison between long and short trades
   - Helps identify directional bias in trading system

8. **Volatility-Adjusted Returns**: Returns adjusted for market volatility
   - Normalizes performance across different market conditions

9. **Recovery Factor**: Total return / Maximum drawdown
   - Measures how effectively a system recovers from drawdowns

10. **Pain Index**: Area under the drawdown curve
    - Comprehensive measure of drawdown magnitude and duration

## Implementing Metrics Validation

The `test_metrics_validation.py` script allows comprehensive metrics validation:

1. **Usage**:
   ```bash
   conda activate Autogen && python tests/test_metrics_validation.py
   ```

2. **Methodology**:
   - Creates synthetic test trades with known outcomes
   - Calculates metrics via three approaches: manual, direct SQL, metrics system
   - Compares results to identify discrepancies

3. **Output**:
   - Detailed comparison table of all metrics
   - Validation summary showing pass/fail rates
   - CSV export of detailed results

4. **Troubleshooting**:
   - ✅ Pass: Metric within 5% of expected value
   - ⚠️ Warning: Metric within 5-15% of expected value
   - ❌ Fail: Metric differs by >15% from expected value
   - ❓ Unknown: Metric not calculated by all methods

## Development Roadmap

1. **Short-term improvements**:
   - Fix any failing metrics in validation tests
   - Implement consistent NULL handling across all metrics
   - Add proper docstrings to all calculation methods

2. **Medium-term additions**:
   - Implement Sortino Ratio and Calmar Ratio
   - Add trade efficiency metrics
   - Improve model score calculations

3. **Long-term enhancements**:
   - Implement comprehensive backtesting metrics
   - Add machine learning based metric optimization
   - Create comparative benchmarking system