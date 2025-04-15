-- Delete trades for specified bots
BEGIN;

-- Count trades before deletion
SELECT 'Before deletion:' as status;
SELECT bot_id, COUNT(*) as trade_count, SUM(CASE WHEN trade_pnl IS NOT NULL THEN trade_pnl ELSE 0 END) as total_pnl
FROM sim_bot_trades
WHERE bot_id IN (2, 3, 102)
GROUP BY bot_id;

-- Delete trades
DELETE FROM sim_bot_trades 
WHERE bot_id IN (2, 3, 102);

-- Confirm deletion
SELECT 'After deletion:' as status;
SELECT bot_id, COUNT(*) as trade_count, SUM(CASE WHEN trade_pnl IS NOT NULL THEN trade_pnl ELSE 0 END) as total_pnl
FROM sim_bot_trades
WHERE bot_id IN (2, 3, 102)
GROUP BY bot_id;

-- Verify remaining trade counts
SELECT 'Remaining trades in system:' as status;
SELECT COUNT(*) as total_remaining_trades FROM sim_bot_trades;

COMMIT;