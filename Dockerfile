FROM node:18-alpine AS base

# Set working directory
WORKDIR /app

# Install dependencies
FROM base AS deps
COPY package.json ./
RUN npm install

# Build the app
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production image
FROM base AS runner
WORKDIR /app
ENV NODE_ENV production

# Copy built assets from builder stage
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/build ./build
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/start-quick.js ./start-quick.js

# Run the app
ENV PORT=8080
EXPOSE 8080

# Set environment variables
ENV USE_MOCK_DATA=true

# Explicitly set NODE_ENV for clarity
ENV NODE_ENV=production

# Start the Express server with additional debugging
CMD ["node", "-e", "console.log('Starting app...'); try { require('./start-quick.js'); } catch (e) { console.error('Error running app:', e); }"]