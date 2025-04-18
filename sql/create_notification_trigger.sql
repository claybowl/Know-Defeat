-- Create notification function
CREATE OR REPLACE FUNCTION notify_bot_metrics_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Construct a JSON payload with the updated data
    PERFORM pg_notify(
        'bot_metrics_channel',
        json_build_object(
            'table', TG_TABLE_NAME,
            'action', TG_OP,
            'bot_id', NEW.bot_id,
            'data', row_to_json(NEW)
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS bot_metrics_change_trigger ON bot_metrics;

-- Create trigger for INSERT and UPDATE operations
CREATE TRIGGER bot_metrics_change_trigger
AFTER INSERT OR UPDATE ON bot_metrics
FOR EACH ROW
EXECUTE FUNCTION notify_bot_metrics_change();

-- Cleanup function for manual execution (if needed)
CREATE OR REPLACE FUNCTION cleanup_bot_metrics_notification()
RETURNS void AS $$
BEGIN
    DROP TRIGGER IF EXISTS bot_metrics_change_trigger ON bot_metrics;
    DROP FUNCTION IF EXISTS notify_bot_metrics_change();
END;
$$ LANGUAGE plpgsql;

-- Usage note:
-- To set up: Just run this script
-- To clean up: SELECT cleanup_bot_metrics_notification(); 