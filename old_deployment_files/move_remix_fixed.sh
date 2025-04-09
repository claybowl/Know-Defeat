#!/bin/bash

# Base directory
BASE_DIR="/mnt/c/Users/clayb/Desktop/CurveAI/Know-Defeat"
TARGET_DIR="$BASE_DIR/old_deployment_files"

# Create target directories
mkdir -p "$TARGET_DIR/remix"
mkdir -p "$TARGET_DIR/app"

# Move root-level Remix files
if [ -f "$BASE_DIR/remix.env.d.ts" ]; then
  mv "$BASE_DIR/remix.env.d.ts" "$TARGET_DIR/"
fi

if [ -f "$BASE_DIR/vite.config.js" ]; then
  mv "$BASE_DIR/vite.config.js" "$TARGET_DIR/"
fi

# Copy the main app directory
cp -r "$BASE_DIR/app/"* "$TARGET_DIR/app/"

# Copy specific files from remix-minimal
REMIX_FILES=(
  ".dockerignore"
  ".gitignore"
  "DEPLOYMENT_GUIDE.md"
  "Dockerfile"
  "deploy.sh"
  "dev.js"
  "full-deploy.sh"
  "package.json"
  "remix.config.js"
  "remix.env.d.ts"
  "server.js"
  "setup-dev.sh"
  "simple.js"
  "test-db-connection.sh"
  "tsconfig.json"
)

for file in "${REMIX_FILES[@]}"; do
  if [ -f "$BASE_DIR/remix-minimal/$file" ]; then
    cp "$BASE_DIR/remix-minimal/$file" "$TARGET_DIR/remix/"
  fi
done

# Copy directories (excluding node_modules, build, .cache)
mkdir -p "$TARGET_DIR/remix/app"
mkdir -p "$TARGET_DIR/remix/db"
mkdir -p "$TARGET_DIR/remix/public"

if [ -d "$BASE_DIR/remix-minimal/app" ]; then
  cp -r "$BASE_DIR/remix-minimal/app/"* "$TARGET_DIR/remix/app/"
fi

if [ -d "$BASE_DIR/remix-minimal/db" ]; then
  cp -r "$BASE_DIR/remix-minimal/db/"* "$TARGET_DIR/remix/db/"
fi

if [ -d "$BASE_DIR/remix-minimal/public" ]; then
  cp -r "$BASE_DIR/remix-minimal/public/"* "$TARGET_DIR/remix/public/"
fi

echo "Remix files have been moved to $TARGET_DIR"