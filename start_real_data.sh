#!/bin/bash

# Start the UI application with REAL database data
echo "Starting Know Defeat UI with REAL database data..."

# Copy the real data environment file
cp .env.real .env

# Start the development server
echo "Starting development server with real database connection..."
npm run dev