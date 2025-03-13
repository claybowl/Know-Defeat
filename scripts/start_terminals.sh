#!/bin/bash
# ================================================
# Step 1: Start the Database
# echo "Starting database..."
# Ensure conda is initialized in this shell (adjust the path to conda.sh as needed)
source ~/anaconda3/etc/profile.d/conda.sh
# conda activate Autogen

# # Note: Use Unix-style paths for Git Bash; "C:\Users\clayb\postgres_data" becomes "/c/Users/clayb/postgres_data"
# pg_ctl -D "/c/Users/clayb/postgres_data" start

# echo "Waiting 10 seconds for the database to become ready..."
# sleep 10

# # ================================================
# # (Optional) Launch psql console in its own window using Mintty
# echo "Launching psql console..."
# mintty -e bash -l -c "conda activate Autogen && psql -U clayb tick_data; exec bash" &

# # ================================================
# # Step 2: Start the IB Controller in its own window
# echo "Starting IB Controller..."
# mintty -e bash -l -c "conda activate Autogen && python src/ib_controller.py; exec bash" &
# echo "Waiting 5 seconds for the IB Controller to initialize..."
# sleep 5

# ================================================
# Step 3: Start all Trading Bots in separate tabs
echo "Starting trading bots (each in a new tab)..."

mintty -t "COIN_long_bot" -e bash -l -c "conda activate Autogen && python src/bots/COIN_long_bot.py; exec bash" &
mintty -t "COIN_short_bot" -e bash -l -c "conda activate Autogen && python src/bots/COIN_short_bot.py; exec bash" &
mintty -t "TSLA_long_bot" -e bash -l -c "conda activate Autogen && python src/bots/TSLA_long_bot.py; exec bash" &
mintty -t "TSLA_short_bot" -e bash -l -c "conda activate Autogen && python src/bots/TSLA_short_bot.py; exec bash" &
mintty -t "COIN_long_bot2" -e bash -l -c "conda activate Autogen && python src/bots/COIN_long_bot2.py; exec bash" &
mintty -t "COIN_short_bot2" -e bash -l -c "conda activate Autogen && python src/bots/COIN_short_bot2.py; exec bash" &
mintty -t "TSLA_long_bot2" -e bash -l -c "conda activate Autogen && python src/bots/TSLA_long_bot2.py; exec bash" &
mintty -t "TSLA_short_bot2" -e bash -l -c "conda activate Autogen && python src/bots/TSLA_short_bot2.py; exec bash" &

echo "All processes started."
read -n 1 -s -r -p "Press any key to exit..."  