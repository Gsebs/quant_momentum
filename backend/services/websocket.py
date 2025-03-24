from fastapi import WebSocket
from typing import Dict, List, Set
import json
import asyncio
from datetime import datetime
from .database import DatabaseService

class WebSocketManager:
    def __init__(self, db_service: DatabaseService):
        self.active_connections: Set[WebSocket] = set()
        self.db_service = db_service
        self.broadcast_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        if not self.broadcast_task:
            self.broadcast_task = asyncio.create_task(self.broadcast_updates())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if not self.active_connections and self.broadcast_task:
            self.broadcast_task.cancel()
            self.broadcast_task = None

    async def broadcast_updates(self):
        while True:
            try:
                # Get latest system status
                status = self.db_service.get_latest_system_status()
                if status:
                    await self.broadcast_message({
                        "type": "system_status",
                        "data": {
                            "status": status.status,
                            "active_trades": status.active_trades,
                            "open_opportunities": status.open_opportunities,
                            "timestamp": status.timestamp.isoformat()
                        }
                    })

                # Get latest opportunities
                opportunities = self.db_service.get_opportunities(limit=10, status="active")
                if opportunities:
                    await self.broadcast_message({
                        "type": "opportunities",
                        "data": [{
                            "id": opp.id,
                            "symbol": opp.symbol,
                            "buy_exchange": opp.buy_exchange,
                            "sell_exchange": opp.sell_exchange,
                            "buy_price": opp.buy_price,
                            "sell_price": opp.sell_price,
                            "price_difference": opp.price_difference,
                            "estimated_profit": opp.estimated_profit,
                            "confidence": opp.confidence,
                            "timestamp": opp.timestamp.isoformat()
                        } for opp in opportunities]
                    })

                # Get latest trades
                trades = self.db_service.get_trades(limit=10)
                if trades:
                    await self.broadcast_message({
                        "type": "trades",
                        "data": [{
                            "id": trade.id,
                            "symbol": trade.symbol,
                            "exchange": trade.exchange,
                            "side": trade.side,
                            "amount": trade.amount,
                            "price": trade.price,
                            "status": trade.status,
                            "profit_loss": trade.profit_loss,
                            "timestamp": trade.timestamp.isoformat()
                        } for trade in trades]
                    })

                # Get performance stats
                stats = self.db_service.get_trading_stats()
                await self.broadcast_message({
                    "type": "performance",
                    "data": stats
                })

                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error in broadcast_updates: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def broadcast_message(self, message: Dict):
        if not self.active_connections:
            return

        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                print(f"Error sending message: {str(e)}")
                disconnected.add(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending personal message: {str(e)}")
            self.disconnect(websocket) 