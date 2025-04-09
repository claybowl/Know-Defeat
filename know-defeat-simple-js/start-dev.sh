#!/bin/bash
# Start both API and UI in development mode

# Start API
echo "Starting API server..."
cd api
npm run dev &
API_PID=$!

# Navigate back to root directory
cd ..

# Start UI
echo "Starting UI development server..."
cd ui
npm install @rollup/rollup-win32-x64-msvc --no-save
npm run dev &
UI_PID=$!

# Function to handle interruption
function cleanup {
  echo "Stopping servers..."
  kill $API_PID
  kill $UI_PID
  exit 0
}

# Trap interruption signal
trap cleanup SIGINT

echo ""
echo "Development servers started:"
echo "API: http://localhost:8080"
echo "UI: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
wait