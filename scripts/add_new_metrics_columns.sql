-- Add new metrics columns to bot_metrics table
BEGIN;

-- Add Sortino ratio column
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS sortino_ratio NUMERIC(12,6);

-- Add Calmar ratio column
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS calmar_ratio NUMERIC(12,6);

-- Add R-multiple column (ratio of average win to average loss)
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS r_multiple NUMERIC(12,6);

-- Add maximum drawdown duration column
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS max_drawdown_duration NUMERIC(20,4);

-- Add recovery factor column
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS recovery_factor NUMERIC(12,6);

-- Add drawdown percent column
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS drawdown_percent NUMERIC(8,4);

-- Add win streak 6 and 7 columns
ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS win_streak_6 NUMERIC(8,4);

ALTER TABLE bot_metrics 
ADD COLUMN IF NOT EXISTS win_streak_7 NUMERIC(8,4);

-- Add timestamp indexes for performance
CREATE INDEX IF NOT EXISTS idx_bot_metrics_bot_id_timestamp 
ON bot_metrics (bot_id, timestamp DESC);

COMMIT;