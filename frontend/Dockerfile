# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Increase Node's memory allocation
ENV NODE_OPTIONS="--max-old-space-size=2048"

# Copy package files
COPY package*.json ./

# Install dependencies with memory optimization flags
RUN npm install --production=false \
    --no-optional \
    --prefer-offline \
    --no-audit \
    --no-fund \
    --quiet

# Install only the specific additional dependencies needed
RUN npm install --no-optional \
    chart.js@3.9.1 \
    chartjs-plugin-zoom@1.2.1 \
    react-chartjs-2@4.3.1 \
    @babel/plugin-proposal-private-property-in-object

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install serve with specific version
RUN npm install -g serve@14.2.1

# Create a non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy build files from builder stage
COPY --from=builder /app/build ./build

# Set proper ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 3000

# Run the app
CMD ["serve", "-s", "build", "-l", "3000"]