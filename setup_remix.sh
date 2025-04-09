#!/bin/bash

# Setup the UI application
echo "Setting up Know Defeat Remix UI..."

# Remove previous installation
echo "Cleaning previous installation..."
rm -rf node_modules package-lock.json node-runner.mjs

# Install dependencies
echo "Installing dependencies with legacy-peer-deps..."
npm install --legacy-peer-deps

echo "Setup complete. Run ./start_ui.sh to start the development server."