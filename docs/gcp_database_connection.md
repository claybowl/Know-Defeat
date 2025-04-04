# GCP Cloud SQL Database Connection Guide

This guide explains how to connect to and test your Google Cloud SQL PostgreSQL database.

## Prerequisites

1. Google Cloud SDK installed and configured
2. Python 3.8+ with required packages:
   - asyncpg
   - python-dotenv
3. Cloud SQL Proxy (for local development)

## Connection Methods

There are two main ways to connect to your Cloud SQL instance:

### 1. Using Cloud SQL Proxy (Recommended for Development)

The Cloud SQL Proxy provides a secure way to connect to your Cloud SQL instance without configuring SSL or a static IP.

#### Setup Cloud SQL Proxy:

1. Run the provided setup script:
   ```bash
   ./scripts/setup_cloud_sql_proxy.sh
   ```

2. This script will:
   - Download the Cloud SQL Proxy if needed
   - Authenticate with Google Cloud
   - Start the proxy on port 5432

3. The proxy will create a local connection to your Cloud SQL instance, allowing you to connect as if the database was running locally.

### 2. Direct Connection (For Production)

For production applications deployed to GCP services like Cloud Run, you can connect directly to the Cloud SQL instance.

## Testing the Connection

### Basic Connection Test

Run the simple connection test script:

```bash
python3 scripts/test_gcp_connection.py
```

This script will:
1. Connect to your database using settings from your `.env` file
2. Run a simple query to verify the connection
3. Report success or failure

### Comprehensive Health Check

For a more detailed analysis of your database:

```bash
python3 scripts/gcp_db_health_check.py
```

This script will:
1. Test database connection
2. Report database size
3. List all tables and their record counts
4. Show recent bot trades
5. Display top-ranked bots
6. Provide database health metrics

## Environment Configuration

All scripts use the following environment variables from a `.env` file:

```
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=tick_data
DB_HOST=127.0.0.1
DB_PORT=5432
CLOUD_SQL_CONNECTION_NAME=know-defeat-trading:us-central1:trading-db
```

## Troubleshooting

### Connection Refused
- Make sure the Cloud SQL Proxy is running
- Verify your GCP credentials with `gcloud auth list`
- Confirm the correct instance name in your connection string

### Authentication Failed
- Double-check your database username and password
- Ensure your service account has the necessary permissions

### Database Not Found
- Verify the database name is correct
- Make sure the database was created during import

## Updating Your Application

To update your application code to use the GCP Cloud SQL:

1. Create a new file `src/cloud_db_connection.py` with:
   ```python
   import asyncpg
   import os
   from dotenv import load_dotenv

   # Load environment variables
   load_dotenv()

   async def create_db_pool():
       return await asyncpg.create_pool(
           user=os.environ.get('DB_USER', 'postgres'),
           password=os.environ.get('DB_PASSWORD', ''),
           database=os.environ.get('DB_NAME', 'tick_data'),
           host=os.environ.get('DB_HOST', 'localhost'),
           port=int(os.environ.get('DB_PORT', '5432'))
       )
   ```

2. Update imports in your application files to use this new connection module