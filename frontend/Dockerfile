# Build stage
FROM node:18-alpine AS builder
WORKDIR /app

# Increase Node's memory limit
ENV NODE_OPTIONS="--max-old-space-size=4096"

# Copy package files
COPY package*.json ./

# Install dependencies with specific flags
RUN npm install --production=false --no-audit --no-fund --prefer-offline

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production
WORKDIR /app

# Install serve
RUN npm install -g serve@14.2.0

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy built files from builder stage
COPY --from=builder /app/build ./build

USER appuser

EXPOSE 3000
CMD ["serve", "-s", "build", "-l", "3000"]