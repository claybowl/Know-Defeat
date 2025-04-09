FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# Explicitly set the environment variables
ENV PORT=8080
ENV NODE_ENV=production

EXPOSE 8080

# Use the JavaScript version directly
CMD ["node", "src/server.js"]