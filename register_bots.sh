#!/bin/bash

# Register bots from the src/bots directory
echo "Registering bots from src/bots directory..."
python src/register_bots.py

# Check if it was successful
if [ $? -eq 0 ]; then
    echo "Bot registration completed successfully!"
else
    echo "Bot registration failed. Check the logs for details."
    exit 1
fi

# Show the number of registered bots
echo "Retrieving bot stats from database..."
# This would ideally use a database query to show the number of bots
# For now, we'll just suggest checking the database
echo "Check database table 'sim_bots' for registered bots."

echo "Done!" 