# Know Defeat - Algorithmic Trading System UI

This is the UI component of the Know Defeat Algorithmic Trading System, built with Remix, Vite, Chakra UI, and ReCharts.

## Tech Stack

- **Remix**: React framework with server-side rendering capabilities
- **Vite**: Fast and efficient build tooling
- **Chakra UI**: Component library with theming support
- **ReCharts**: React charting library for visualizations
- **TypeScript**: Type safety for robust development
- **PostgreSQL**: Database for trading and bot data

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- PostgreSQL running on your system

### Installation

1. Clone the repository
```bash
git clone https://github.com/claybowl/Know-Defeat.git
cd Know-Defeat
```

2. Install dependencies
```bash
npm install
```

3. Create a `.env` file with the following variables (adjust as needed):
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tick_data
DB_USER=clayb
DB_PASSWORD=musicman
```

### Development

Run the development server:
```bash
npm run dev
```

The application will be available at http://localhost:3000.

### Building for Production

Build the application:
```bash
npm run build
```

Preview the production build:
```bash
npm run start
```

## Project Structure

- `/app`: Main application code
  - `/components`: Reusable UI components
    - `/bot`: Bot-related components
    - `/charts`: Data visualization components
    - `/dashboard`: Dashboard UI components
    - `/layout`: Layout components (header, footer, etc.)
  - `/routes`: Pages and nested routes
    - `/_index.tsx`: Landing page
    - `/dashboard.tsx`: Main dashboard
    - `/bots`: Bot management pages
    - `/trades`: Trade viewing pages
    - `/metrics.tsx`: Performance metrics page
    - `/settings.tsx`: Application settings
    - `/allocation`: Fund allocation pages
  - `/lib`: Utilities and server-side code
    - `api.server.js`: API functions
    - `auth.server.js`: Authentication logic
    - `db.server.js`: Database connection
    - `theme.js`: Chakra UI theme
- `/public`: Static assets

## Features

- Real-time dashboard with trade metrics
- Bot management interface
- Trade history visualization
- Performance metrics calculations
- Fund allocation charts
- Bot parameter management

## Database Integration

The application connects to a PostgreSQL database with the following tables:
- `sim_bots`: Bot configuration data
- `sim_bot_trades`: Trading activity
- `bot_metrics`: Performance metrics
- `variable_weights`: Ranking system weights

If the database is not available, the application falls back to mock data for development purposes.

## Charts and Visualizations

The application includes several chart types:
- Trade activity charts
- Fund allocation charts
- Performance metric visualizations
- Bot parameter radar charts
- Trade history charts

## Authentication

A placeholder authentication system is included, which can be expanded to a full authentication system as needed.

## UI Navigation

- **Dashboard**: Overview of system performance and statistics
- **Bots**: Bot management and detailed performance
- **Trades**: Active trades and trade history
- **Metrics**: Detailed system-wide metrics
- **Settings**: System configuration
- **Allocation**: Fund allocation visualization

## Development Notes

- The application's theme can be customized in `app/lib/theme.js`
- Mock data is provided for development in `app/lib/db.server.js`
- The application uses Chakra UI for all components
- ReCharts is used for all data visualizations
- Remix's file-based routing is used for all pages