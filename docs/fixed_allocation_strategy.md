# Fixed 10% Allocation Strategy Implementation

## Overview

The trading system now uses a fixed allocation strategy where each active trade receives exactly 10% of the total fund ($2,000 out of the $20,000 total). This implementation replaces the previous proportional allocation system with a simpler, more predictable approach.

## Key Features

1. **Fixed Trade Allocation**:
   - Each active trade receives exactly 10% of the total fund ($2,000)
   - Maximum of 10 concurrent trades (100% allocation)
   - Total fund size is fixed at $20,000

2. **Rank-Based Trade Prioritization**:
   - Highest-ranked bots get priority for trading slots
   - Lower-ranked bots can trade if spots are available
   - Higher-ranked bots can "bump" lower-ranked bots if needed

3. **Dynamic Trade Management**:
   - When a bot wants to trade, it checks if there are available slots
   - If all slots are filled, it checks if it outranks any currently trading bots
   - If it outranks a trading bot, it can close that bot's trade and take its place

## Implementation Details

### Fund Allocation Logic

```python
async def get_fund_allocation(self, total_funds=20000):
    """
    Calculate fund allocation based on fixed 10% per trade strategy.
    """
    # Get active trades ordered by rank score
    active_trades = await self.trade_manager.get_active_trades()
    
    # Fixed allocation per trade (10% of total funds)
    fixed_trade_amount = total_funds / 10
    
    # Calculate allocations for active trades (each gets 10%)
    for trade in active_trades:
        allocations.append({
            'bot_id': bot_id,
            'ticker': ticker,
            'rank_score': rank_score,
            'rank': rank,
            'trade_id': trade['trade_id'],
            'allocation_amount': fixed_trade_amount,
            'allocation_percentage': 10.0
        })
```

### Trade Initiation Logic

```python
async def initiate_trade(self, bot_id, ticker, entry_price, trade_direction, trade_size=None):
    """
    Initiate a new trade with the dynamic allocation logic using fixed trade size.
    """
    # Set default trade size to fixed 10% of $20,000 total funds
    if trade_size is None:
        trade_size = 2000.0  # Fixed at 10% of $20,000
    
    # Check if this bot can open a trade
    can_trade, needs_to_close, lowest_trade = await self.can_open_new_trade(bot_id)
    
    # If needed, close the lowest-ranked trade to make room
    if needs_to_close:
        closed_trade = await self.close_lowest_ranked_trade()
```

### Trade Replacement Logic

```python
async def can_open_new_trade(self, bot_id):
    """
    Determine if a bot can open a new trade based on current portfolio allocation.
    """
    # If we have fewer than max trades, allow new trade
    if len(active_trades) < self.max_active_trades:
        return (True, False, None)
    
    # If we're at max trades, check if this bot outranks the lowest-ranked active trade
    lowest_ranked_trade = active_trades[-1]
    
    if bot_rank > lowest_ranked_trade['rank_score']:
        # This bot outranks the lowest active trade, so we can close that one
        return (True, True, lowest_ranked_trade)
```

## Benefits of the New Strategy

1. **Predictable Position Sizing**: Each trade has the same fixed size ($2,000), making risk management simpler and more consistent.

2. **Full Capital Utilization**: The system aims to keep all capital deployed when possible by filling all 10 trading slots.

3. **Optimal Bot Selection**: The ranking system ensures that the highest-performing bots get priority for trading, maximizing overall system performance.

4. **Dynamic Adjustments**: As bot rankings change, the system automatically adjusts which bots can trade, ensuring that the best strategies are always in use.

5. **Clear Slot Management**: The 10-slot system makes it easy to understand how much of the fund is deployed at any given time.

## Usage Examples

### Scenario 1: Some Trading Slots Available
If only 7 out of 10 slots are filled, any bot can enter a trade, with priority given to higher-ranked bots.

### Scenario 2: All Slots Filled, Higher-Ranked Bot Wants to Trade
If all 10 slots are filled and the 5th ranked bot (not currently trading) wants to enter a trade, it will:
1. Identify the lowest-ranked active bot (e.g., 24th ranked)
2. Close that bot's trade
3. Enter its own trade in that slot

### Scenario 3: All Slots Filled, Lower-Ranked Bot Wants to Trade
If all 10 slots are filled with bots ranked 1-10, and the 15th ranked bot wants to trade:
- It cannot enter a trade as it doesn't outrank any active bots
- It must wait until a trading slot becomes available

## Testing the Implementation

The system has been tested to verify:
1. Each active trade receives exactly $2,000 (10% of total funds)
2. The trade replacement logic works correctly
3. Fund allocation corresponds to the actual trades in the system
4. Bot rankings properly influence trading opportunities