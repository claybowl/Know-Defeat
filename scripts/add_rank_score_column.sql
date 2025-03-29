-- Script to add rank_score column to the bot_metrics table if it doesn't exist

-- First check if the column exists
DO $$ 
BEGIN
    -- Check if rank_score column already exists
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'bot_metrics' 
        AND column_name = 'rank_score'
    ) THEN
        -- Add the rank_score column
        ALTER TABLE bot_metrics 
        ADD COLUMN rank_score NUMERIC(10,4) NOT NULL DEFAULT 0;
        
        -- Add an index on the rank_score column for faster sorting
        CREATE INDEX idx_bot_metrics_rank_score ON bot_metrics(rank_score);
        
        RAISE NOTICE 'rank_score column added successfully to bot_metrics table';
    ELSE
        RAISE NOTICE 'rank_score column already exists in bot_metrics table';
    END IF;
END $$;

-- Update existing records to calculate a simple rank score based on existing metrics
-- This is a simple example - you may want to adjust the formula based on your specific needs
UPDATE bot_metrics
SET rank_score = 
    (
        -- Win rate contributes 30%
        (CASE WHEN win_rate IS NOT NULL THEN win_rate ELSE 0 END) * 0.3 + 
        
        -- Profit factor contributes 30% (normalized to roughly 0-1 range)
        (CASE WHEN profit_factor IS NOT NULL THEN LEAST(profit_factor / 3, 1) ELSE 0 END) * 0.3 +
        
        -- Sharpe ratio contributes 20% (normalized to roughly 0-1 range)
        (CASE WHEN sharpe_ratio IS NOT NULL THEN LEAST(sharpe_ratio / 2, 1) ELSE 0 END) * 0.2 +
        
        -- Reverse of max_drawdown contributes 10% (lower drawdown is better)
        (CASE WHEN max_drawdown IS NOT NULL THEN 1 - LEAST(ABS(max_drawdown) / 1000, 1) ELSE 0 END) * 0.1 +
        
        -- Expectancy contributes 10% (normalized to roughly 0-1 range)
        (CASE WHEN expectancy IS NOT NULL THEN LEAST(expectancy / 0.5, 1) ELSE 0 END) * 0.1
    );

-- Show the updated bot_metrics with rank_score
SELECT bot_id, total_trades, win_rate, profit_factor, sharpe_ratio, max_drawdown, rank_score
FROM bot_metrics
ORDER BY rank_score DESC;