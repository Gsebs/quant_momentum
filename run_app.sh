#!/bin/bash

# Start the Flask backend
echo "Starting Flask backend..."
python app.py &

# Wait a bit for the backend to start
sleep 2

# Start the React frontend
echo "Starting React frontend..."
cd frontend && npm start 