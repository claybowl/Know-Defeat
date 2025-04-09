# Know-Defeat Deployment Guide

This guide explains how to deploy the Know-Defeat trading system to Google Cloud Platform, connecting it to your PostgreSQL database in Cloud SQL.

## Deployment Options

You can deploy either:
1. **API-only mode**: Just the API endpoints with a simple dashboard
2. **Full UI mode**: Complete Remix UI with all frontend features

### API-Only Mode (Default)

This is the default mode and requires no additional setup. Simply run the deployment script:

```bash
./deploy.sh
```

### Full UI Mode

To deploy with the complete Remix UI:

1. Set up the development environment:
   ```bash
   ./setup-dev.sh
   ```

2. Install dev dependencies:
   ```bash
   npm install -D @remix-run/dev @types/react @types/react-dom typescript
   ```

3. Build the Remix application:
   ```bash
   npm run build
   ```

4. Deploy with the built UI:
   ```bash
   ./deploy.sh
   ```

## Prerequisites

1. Google Cloud SDK installed and configured (`gcloud` command available)
2. Access to the Cloud SQL instance
3. Database credentials

## Testing Database Connection

Before deployment, test your database connection:

```bash
# Test using simulated Cloud SQL socket
npm run test:db

# Or test using Cloud SQL Proxy (for local development)
npm run test:db:proxy
```

This will verify that your database credentials work and that the required tables exist.

## Deployment Options

### 1. Deploy with Mock Data (for Testing)

If you want to test the deployment without connecting to a real database:

```bash
# Run the deployment script and select 'no' when asked about real database
./deploy.sh
```

This will deploy using the mock data included in the application.

### 2. Deploy with Real Database Connection

To deploy with a connection to your Cloud SQL database:

```bash
# Run the deployment script and select 'yes' when asked about real database
./deploy.sh
```

You will be asked for your database password, which will be securely stored in Google Secret Manager.

## Checking Deployment Status

After deployment, you can check the status and logs:

```bash
# View logs from Cloud Run
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=know-defeat-api" --limit 50

# Check service details
gcloud run services describe know-defeat-api --region us-central1
```

## Database Migration Plan

To migrate your complete database to Google Cloud SQL:

1. Export your local database:
   ```bash
   pg_dump -U your_user -d tick_data > tick_data_backup.sql
   ```

2. Import to Cloud SQL:
   ```bash
   gcloud sql import sql trading-db gs://your-bucket/tick_data_backup.sql
   ```

   Or use Cloud SQL proxy to import directly:
   ```bash
   cat tick_data_backup.sql | psql -h 127.0.0.1 -U postgres -d tick_data
   ```

## UI Enhancement Plan

Now that the basic API is working, the next steps for enhancing the UI are:

1. Add more charts and visualizations to the dashboard
2. Create dedicated pages for:
   - Bot management
   - Trade monitoring
   - Metrics analysis
   - System configuration

3. Connect these pages to the existing API endpoints

## Troubleshooting

### Common Issues

- **Connection Refused**: Ensure your Cloud SQL instance is running and the connection name is correct.
- **Authentication Failed**: Verify your database username and password.
- **Tables Not Found**: Make sure you've imported your database schema.
- **Cloud Run Error**: Check the logs for specific error messages.

### Checking Container Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=know-defeat-api AND severity>=ERROR" --limit 20
```

### Testing Database Connection Directly

You can test your database connection directly using the Cloud SQL Proxy:

```bash
# Install Cloud SQL Proxy if needed
curl -o cloud_sql_proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.6.0/cloud-sql-proxy.linux.amd64
chmod +x cloud_sql_proxy

# Start proxy
./cloud_sql_proxy know-defeat-trading:us-central1:trading-db &

# Connect with psql
psql -h 127.0.0.1 -U postgres -d tick_data
```

## Next Steps

After successful deployment with real database connection:

1. Enhance the UI based on the available data
2. Test trading bot metrics display
3. Integrate real-time data capabilities
4. Implement user authentication
5. Add admin controls for bot management

## Help and Support

If you encounter issues with deployment:

1. Check the Cloud Run logs
2. Verify database credentials
3. Try deploying with mock data first to isolate the issue
4. Consider using the test-app deployment for simpler troubleshooting