#!/bin/bash

# Configuration
BACKUP_DIR="/backup"
DB_NAME="trading_db"
DB_USER="user"
DB_PASSWORD="password"
DB_HOST="db"

# Check if backup file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 backup_20240101_120000.sql.gz"
    exit 1
fi

BACKUP_FILE="$BACKUP_DIR/$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Stop the application
echo "Stopping application..."
docker-compose stop app

# Drop existing database
echo "Dropping existing database..."
PGPASSWORD=$DB_PASSWORD dropdb -h $DB_HOST -U $DB_USER $DB_NAME

# Create new database
echo "Creating new database..."
PGPASSWORD=$DB_PASSWORD createdb -h $DB_HOST -U $DB_USER $DB_NAME

# Restore from backup
echo "Restoring from backup..."
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER $DB_NAME
else
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER $DB_NAME < $BACKUP_FILE
fi

# Check if restore was successful
if [ $? -eq 0 ]; then
    echo "Database restored successfully"
    
    # Start the application
    echo "Starting application..."
    docker-compose start app
    
    echo "Recovery completed successfully"
else
    echo "Recovery failed!"
    exit 1
fi 