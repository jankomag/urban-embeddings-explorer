# Build stage
FROM node:14 AS build

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Install additional dependencies
RUN npm install chart.js@3.9.1 chartjs-plugin-zoom@1.2.1 react-chartjs-2@4.3.1 @babel/plugin-proposal-private-property-in-object

# Copy the current directory contents into the container
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM node:14-alpine

WORKDIR /app

# Install serve
RUN npm install -g serve

# Copy build files from build stage
COPY --from=build /app/build ./build

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run the app when the container launches
CMD ["serve", "-s", "build", "-l", "3000"]