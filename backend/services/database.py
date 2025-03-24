from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..models.database import Trade, Opportunity, Performance, SystemStatus

class DatabaseService:
    def __init__(self, db: Session):
        self.db = db

    # Trade operations
    def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
        trade = Trade(**trade_data)
        self.db.add(trade)
        self.db.commit()
        self.db.refresh(trade)
        return trade

    def get_trades(
        self,
        skip: int = 0,
        limit: int = 100,
        symbol: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Trade]:
        query = self.db.query(Trade)
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        if status:
            query = query.filter(Trade.status == status)
        return query.offset(skip).limit(limit).all()

    def update_trade(self, trade_id: int, trade_data: Dict[str, Any]) -> Optional[Trade]:
        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if trade:
            for key, value in trade_data.items():
                setattr(trade, key, value)
            self.db.commit()
            self.db.refresh(trade)
        return trade

    # Opportunity operations
    def create_opportunity(self, opportunity_data: Dict[str, Any]) -> Opportunity:
        opportunity = Opportunity(**opportunity_data)
        self.db.add(opportunity)
        self.db.commit()
        self.db.refresh(opportunity)
        return opportunity

    def get_opportunities(
        self,
        skip: int = 0,
        limit: int = 100,
        symbol: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Opportunity]:
        query = self.db.query(Opportunity)
        if symbol:
            query = query.filter(Opportunity.symbol == symbol)
        if status:
            query = query.filter(Opportunity.status == status)
        return query.offset(skip).limit(limit).all()

    def update_opportunity(self, opportunity_id: int, opportunity_data: Dict[str, Any]) -> Optional[Opportunity]:
        opportunity = self.db.query(Opportunity).filter(Opportunity.id == opportunity_id).first()
        if opportunity:
            for key, value in opportunity_data.items():
                setattr(opportunity, key, value)
            self.db.commit()
            self.db.refresh(opportunity)
        return opportunity

    # Performance operations
    def create_performance(self, performance_data: Dict[str, Any]) -> Performance:
        performance = Performance(**performance_data)
        self.db.add(performance)
        self.db.commit()
        self.db.refresh(performance)
        return performance

    def get_performance_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Performance]:
        query = self.db.query(Performance)
        if start_time:
            query = query.filter(Performance.timestamp >= start_time)
        if end_time:
            query = query.filter(Performance.timestamp <= end_time)
        return query.order_by(Performance.timestamp.desc()).limit(limit).all()

    # System status operations
    def create_system_status(self, status_data: Dict[str, Any]) -> SystemStatus:
        status = SystemStatus(**status_data)
        self.db.add(status)
        self.db.commit()
        self.db.refresh(status)
        return status

    def get_latest_system_status(self) -> Optional[SystemStatus]:
        return self.db.query(SystemStatus).order_by(SystemStatus.timestamp.desc()).first()

    def get_system_status_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SystemStatus]:
        query = self.db.query(SystemStatus)
        if start_time:
            query = query.filter(SystemStatus.timestamp >= start_time)
        if end_time:
            query = query.filter(SystemStatus.timestamp <= end_time)
        return query.order_by(SystemStatus.timestamp.desc()).limit(limit).all()

    # Analytics operations
    def get_trading_stats(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        query = self.db.query(Trade)
        if start_time:
            query = query.filter(Trade.timestamp >= start_time)
        if end_time:
            query = query.filter(Trade.timestamp <= end_time)

        trades = query.all()
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_profit_loss": 0,
                "average_profit_loss": 0
            }

        winning_trades = len([t for t in trades if t.profit_loss and t.profit_loss > 0])
        total_profit_loss = sum(t.profit_loss or 0 for t in trades)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": winning_trades / total_trades,
            "total_profit_loss": total_profit_loss,
            "average_profit_loss": total_profit_loss / total_trades
        } 