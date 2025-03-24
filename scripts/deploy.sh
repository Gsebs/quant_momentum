#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment process...${NC}"

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo -e "${RED}Heroku CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if logged into Heroku
if ! heroku auth:whoami &> /dev/null; then
    echo -e "${RED}Not logged into Heroku. Please run 'heroku login' first.${NC}"
    exit 1
fi

# Create Heroku app if it doesn't exist
if ! heroku apps:info quant-momentum-hft &> /dev/null; then
    echo -e "${GREEN}Creating Heroku app...${NC}"
    heroku create quant-momentum-hft
fi

# Set up Heroku container stack
echo -e "${GREEN}Setting up Heroku container stack...${NC}"
heroku stack:set container -a quant-momentum-hft

# Add PostgreSQL if not already added
if ! heroku addons:info postgresql -a quant-momentum-hft &> /dev/null; then
    echo -e "${GREEN}Adding PostgreSQL addon...${NC}"
    heroku addons:create heroku-postgresql:hobby-dev -a quant-momentum-hft
fi

# Set environment variables
echo -e "${GREEN}Setting up environment variables...${NC}"
echo "Please enter your Binance API key:"
read -s BINANCE_API_KEY
echo "Please enter your Binance API secret:"
read -s BINANCE_API_SECRET
echo "Please enter your Coinbase API key:"
read -s COINBASE_API_KEY
echo "Please enter your Coinbase API secret:"
read -s COINBASE_API_SECRET

heroku config:set \
    BINANCE_API_KEY=$BINANCE_API_KEY \
    BINANCE_API_SECRET=$BINANCE_API_SECRET \
    COINBASE_API_KEY=$COINBASE_API_KEY \
    COINBASE_API_SECRET=$COINBASE_API_SECRET \
    ENVIRONMENT=production \
    -a quant-momentum-hft

# Deploy to Heroku
echo -e "${GREEN}Deploying to Heroku...${NC}"
git push heroku main

# Deploy frontend to Netlify
echo -e "${GREEN}Building frontend for production...${NC}"
cd frontend
npm install
npm run build

echo -e "${GREEN}Deploying to Netlify...${NC}"
if ! command -v netlify &> /dev/null; then
    echo -e "${RED}Netlify CLI is not installed. Installing...${NC}"
    npm install -g netlify-cli
fi

# Deploy to Netlify
netlify deploy --prod --dir=build

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "Backend URL: https://quant-momentum-hft.herokuapp.com"
echo -e "Check Netlify dashboard for frontend URL" 