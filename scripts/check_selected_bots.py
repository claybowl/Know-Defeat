import asyncio
import asyncpg
import sys
from decimal import Decimal

async def check_specific_bots():
    try:
        print("Connecting to database to check bots 1, 5, 7, and 103...")
        # Connect to the database
        conn = await asyncpg.connect(
            user='clayb',
            password='musicman',
            database='tick_data',
            host='localhost'
        )
        
        bot_ids = [1, 5, 7, 103]
        
        # Check sim_bots table
        print("\n=== Bot Configuration from sim_bots ===")
        bot_configs = await conn.fetch("""
            SELECT * FROM sim_bots WHERE bot_id = ANY($1)
        """, bot_ids)
        
        if not bot_configs:
            print("No records found in sim_bots for these bot IDs!")
        else:
            for bot in bot_configs:
                print(f"Bot {bot['bot_id']}: {bot.get('name', 'N/A')} - {bot.get('ticker', 'N/A')} - {bot.get('algorithm_type', 'N/A')}")
                print(f"  Trade Direction: {bot.get('trade_direction', 'N/A')}")
                print(f"  Position Size: ${bot.get('position_size', 'N/A')}")
                print(f"  Active: {bot.get('is_active', 'N/A')}")
                print("  ---")
        
        # Check bot_metrics table
        print("\n=== Latest Bot Metrics from bot_metrics ===")
        bot_metrics = await conn.fetch("""
            SELECT DISTINCT ON (bot_id) * 
            FROM bot_metrics 
            WHERE bot_id = ANY($1)
            ORDER BY bot_id, timestamp DESC
        """, bot_ids)
        
        if not bot_metrics:
            print("No records found in bot_metrics for these bot IDs!")
        else:
            for metrics in bot_metrics:
                bot_id = metrics['bot_id']
                print(f"Bot {bot_id} metrics (timestamp: {metrics.get('timestamp', 'N/A')}):")
                
                # Metrics to check
                key_metrics = [
                    'avg_win_rate', 'win_rate', 'profit_factor', 'total_pnl', 
                    'one_hour_performance', 'one_day_performance',
                    'current_rank', 'sharpe_ratio'
                ]
                
                for metric in key_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        # Format percentages
                        if metric in ['avg_win_rate', 'win_rate', 'one_hour_performance', 'one_day_performance']:
                            print(f"  {metric}: {value}%")
                        # Format monetary values
                        elif metric in ['total_pnl']:
                            print(f"  {metric}: ${value}")
                        else:
                            print(f"  {metric}: {value}")
                
                # Check for suspiciously high values
                warnings = []
                
                # Either avg_win_rate or win_rate could be used
                win_rate = metrics.get('avg_win_rate', metrics.get('win_rate', 0))
                if win_rate is not None and float(win_rate or 0) > 95:
                    warnings.append(f"Suspiciously high win rate: {win_rate}%")
                
                profit_factor = metrics.get('profit_factor')
                if profit_factor is not None and float(profit_factor or 0) > 50:
                    warnings.append(f"Extreme profit factor: {profit_factor}")
                
                if warnings:
                    print("  WARNINGS:")
                    for warning in warnings:
                        print(f"  - {warning}")
                
                print("  ---")
        
        # Check bot_rankings table
        print("\n=== Bot Rankings from bot_rankings ===")
        rankings = await conn.fetch("""
            SELECT * FROM bot_rankings WHERE bot_id = ANY($1)
        """, bot_ids)
        
        if not rankings:
            print("No records found in bot_rankings for these bot IDs!")
        else:
            for ranking in rankings:
                print(f"Bot {ranking['bot_id']}:")
                print(f"  Rank: {ranking.get('rank', 'N/A')}")
                print(f"  Rank Score: {ranking.get('rank_score', 'N/A')}")
                print(f"  Is Active: {ranking.get('is_active', 'N/A')}")
                print(f"  Last Updated: {ranking.get('timestamp', 'N/A')}")
                print("  ---")
        
        # Check trades in sim_bot_trades
        print("\n=== Recent Trades from sim_bot_trades ===")
        trades = await conn.fetch("""
            SELECT bot_id, trade_id, entry_time, exit_time, trade_status, 
                   entry_price, exit_price, trade_direction, trade_pnl
            FROM sim_bot_trades
            WHERE bot_id = ANY($1)
            ORDER BY bot_id, entry_time DESC
            LIMIT 50
        """, bot_ids)
        
        # Group trades by bot
        trades_by_bot = {}
        for trade in trades:
            bot_id = trade['bot_id']
            if bot_id not in trades_by_bot:
                trades_by_bot[bot_id] = []
            trades_by_bot[bot_id].append(trade)
        
        if not trades:
            print("No trades found in sim_bot_trades for these bot IDs!")
        else:
            for bot_id in bot_ids:
                bot_trades = trades_by_bot.get(bot_id, [])
                
                if not bot_trades:
                    print(f"Bot {bot_id}: No trades found")
                    continue
                
                print(f"Bot {bot_id} trades:")
                print(f"  Total trades found: {len(bot_trades)}")
                
                # Count trades by status
                open_count = sum(1 for t in bot_trades if t['trade_status'] == 'open')
                closed_count = sum(1 for t in bot_trades if t['trade_status'] == 'closed')
                pending_count = sum(1 for t in bot_trades if t['trade_status'] == 'pending_exit')
                
                print(f"  Open trades: {open_count}")
                print(f"  Closed trades: {closed_count}")
                print(f"  Pending exit: {pending_count}")
                
                # Calculate trade statistics from closed trades
                closed_trades = [t for t in bot_trades if t['trade_status'] == 'closed']
                if closed_trades:
                    win_count = sum(1 for t in closed_trades if t.get('trade_pnl') is not None and float(t['trade_pnl']) > 0)
                    loss_count = sum(1 for t in closed_trades if t.get('trade_pnl') is not None and float(t['trade_pnl']) <= 0)
                    pnl_values = [float(t['trade_pnl']) for t in closed_trades if t.get('trade_pnl') is not None]
                    
                    # Calculate win rate and total PnL
                    win_rate = (win_count / len(closed_trades)) * 100 if closed_trades else 0
                    total_pnl = sum(pnl_values)
                    
                    print(f"  Win/Loss: {win_count}/{loss_count}")
                    print(f"  Calculated Win Rate: {win_rate:.2f}%")
                    print(f"  Calculated Total PnL: ${total_pnl:.2f}")
                    
                    # Compare with reported metrics if available
                    bot_metric = next((m for m in bot_metrics if m['bot_id'] == bot_id), None)
                    if bot_metric:
                        reported_win_rate = bot_metric.get('avg_win_rate') or bot_metric.get('win_rate')
                        reported_pnl = bot_metric.get('total_pnl')
                        
                        if reported_win_rate is not None:
                            win_rate_diff = abs(float(reported_win_rate) - win_rate)
                            if win_rate_diff > 10:
                                print(f"  WARNING: Win rate discrepancy - reported {reported_win_rate}% vs calculated {win_rate:.2f}%")
                        
                        if reported_pnl is not None:
                            try:
                                pnl_diff = abs(float(reported_pnl) - total_pnl)
                                if pnl_diff > 100:
                                    print(f"  WARNING: PnL discrepancy - reported ${reported_pnl} vs calculated ${total_pnl:.2f}")
                            except (ValueError, TypeError):
                                print(f"  WARNING: Could not compare PnL values - reported: {reported_pnl}, calculated: {total_pnl:.2f}")
                
                # Analyze most recent trades
                print("  Recent trades:")
                for i, trade in enumerate(bot_trades[:5]):  # Show 5 most recent trades
                    status = trade['trade_status']
                    direction = trade['trade_direction']
                    entry_time = trade['entry_time']
                    entry_price = trade['entry_price']
                    
                    trade_info = f"    {i+1}. {status.upper()} {direction} trade from {entry_time}, entry: ${entry_price}"
                    
                    if status == 'closed':
                        exit_time = trade['exit_time']
                        exit_price = trade['exit_price']
                        pnl = trade.get('trade_pnl')
                        
                        if all(v is not None for v in [exit_time, exit_price, pnl]):
                            result = "WIN" if float(pnl) > 0 else "LOSS"
                            trade_info += f", exit: ${exit_price}, PnL: ${pnl} ({result})"
                    
                    print(trade_info)
                
                print("  ---")
        
        await conn.close()
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# Run the async function
if __name__ == "__main__":
    asyncio.run(check_specific_bots()) 