# HFT Latency Arbitrage Trading System

A high-frequency trading system focused on latency arbitrage opportunities across multiple exchanges.

## Features

- Real-time market data processing
- Latency arbitrage detection
- Automated order execution
- Risk management system
- Real-time monitoring dashboard
- Performance analytics

## Project Structure

```
quant_momentum/
├── backend/
│   ├── api/
│   ├── core/
│   ├── models/
│   └── services/
├── frontend/
│   ├── src/
│   └── public/
├── tests/
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret_key
```

4. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

5. Start the frontend development server:
```bash
cd frontend
npm install
npm start
```

## Configuration

The system can be configured through the frontend dashboard or by modifying the configuration files in the `backend/config` directory.

## Risk Management

The system includes built-in risk management features:
- Position limits
- Loss limits
- Maximum drawdown controls
- Automated stop-loss mechanisms

## Monitoring

Access the monitoring dashboard at `http://localhost:3000` to view:
- Real-time trading activity
- Performance metrics
- Risk indicators
- System health status

## License

MIT License
