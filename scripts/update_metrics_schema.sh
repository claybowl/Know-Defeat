#!/bin/bash

echo "Updating bot metrics schema with new columns..."

# Get database credentials
DB_USER="clayb"
DB_NAME="tick_data"
DB_HOST="localhost"

# Run the SQL script
psql -U $DB_USER -h $DB_HOST -d $DB_NAME -f scripts/add_new_metrics_columns.sql

echo "Schema update complete."