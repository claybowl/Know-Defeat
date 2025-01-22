from datetime import datetime
import pytz

def is_market_hours():
    """Check if current time is within regular market hours (9:31 AM - 3:45 PM ET)"""
    et_timezone = pytz.timezone('US/Eastern')
    current_time = datetime.now(et_timezone)

    # Only trade during market hours on weekdays
    if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    market_start = current_time.replace(hour=9, minute=31, second=0, microsecond=0)
    market_end = current_time.replace(hour=15, minute=45, second=0, microsecond=0)

    return market_start <= current_time <= market_end

def get_current_minute_start():
    """Get the start of the current minute in Eastern Time"""
    et_timezone = pytz.timezone('US/Eastern')
    current_time = datetime.now(et_timezone)
    return current_time.replace(second=0, microsecond=0)
