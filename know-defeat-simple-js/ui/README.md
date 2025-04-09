# Know-Defeat Trading System UI

This is the React-based UI for the Know-Defeat algorithmic trading system.

## Features

- Dashboard with performance metrics
- Bot management and details
- Trade monitoring
- Performance metrics visualization
- Fund allocation management

## Tech Stack

- React 19
- TypeScript
- Material UI 5
- React Query
- React Router
- Recharts for data visualization
- Vite for build tooling

## Development

To run the development server:

```bash
npm install
npm run dev
```

## Building for Production

To build for production:

```bash
npm run build
```

The output will be in the `dist` directory.

## Configuration

Environment variables can be set in `.env` files:

- `VITE_API_URL` - API endpoint URL

## Deployment Options

### Option 1: Manual Deployment

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

### Option 2: Cloud Build

```bash
# Trigger a Cloud Build deployment
gcloud builds submit --config=cloudbuild.yaml
```

### Option 3: Docker

```bash
# Build the Docker image locally
docker build -t know-defeat-ui .

# Run the container
docker run -p 8080:80 know-defeat-ui
```

## Troubleshooting

If you encounter issues with Material UI version compatibility, you can use the downgrade script:

```bash
chmod +x downgrade-mui.sh
./downgrade-mui.sh
```

This will install Material UI 5.x which is compatible with the current Grid implementation.

## API Integration

The UI expects the following API endpoints:

- `/api/bots` - List all bots
- `/api/bots/:id` - Get bot details
- `/api/trades` - List all trades
- `/api/trades/open` - List open trades
- `/api/metrics` - Get performance metrics
- `/api/dashboard` - Get dashboard data
- `/api/allocation` - Get fund allocation data