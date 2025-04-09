# Build stage
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./

# Set NODE_ENV to production to avoid platform-specific dependencies
ENV NODE_ENV=production

RUN npm install --legacy-peer-deps

COPY . .

# Use NODE_ENV=production to avoid platform-specific dependencies
RUN NODE_OPTIONS=--legacy-peer-deps npm run build

# Production stage
FROM nginx:alpine

# Copy built files from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]