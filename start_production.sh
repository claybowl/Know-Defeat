#!/bin/bash
# Start PostgreSQL if not running
pg_ctl -D "C:/Users/clayb/postgres_data" status || pg_ctl -D "C:/Users/clayb/postgres_data" start

# Run with production server
export ENV=production
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
