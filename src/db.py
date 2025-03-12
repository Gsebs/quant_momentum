"""
Database module for storing historical data and momentum metrics.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumDB:
    def __init__(self, db_path: str):
        """Initialize database connection."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self._init_db()
            logger.info(f"Connected to database at {db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def _init_db(self):
        """Initialize database tables."""
        try:
            cursor = self.conn.cursor()
            
            # Drop existing tables and indices
            cursor.execute("DROP TABLE IF EXISTS historical_data")
            cursor.execute("DROP TABLE IF EXISTS momentum_metrics")
            
            # Create historical data table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                ticker TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, timestamp)
            )
            """)
            
            # Create momentum metrics table with all indicators, ranks, and position sizes
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS momentum_metrics (
                ticker TEXT,
                timestamp TEXT,
                last_price REAL,
                avg_volume REAL,
                composite_score REAL,
                ml_score REAL,
                enhanced_score REAL,
                position_size REAL,
                momentum_1m REAL,
                momentum_3m REAL,
                momentum_6m REAL,
                momentum_12m REAL,
                momentum_1m_rank REAL,
                momentum_3m_rank REAL,
                momentum_6m_rank REAL,
                momentum_12m_rank REAL,
                rsi REAL,
                macd REAL,
                signal REAL,
                histogram REAL,
                upper REAL,
                middle REAL,
                lower REAL,
                volatility REAL,
                roc_5 REAL,
                roc_10 REAL,
                roc_20 REAL,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                volume_sma REAL,
                volume_ratio REAL,
                rsi_rank REAL,
                macd_rank REAL,
                volatility_rank REAL,
                risk_total_return REAL,
                risk_annualized_return REAL,
                risk_annualized_volatility REAL,
                risk_sharpe_ratio REAL,
                risk_sortino_ratio REAL,
                risk_max_drawdown REAL,
                risk_drawdown_duration REAL,
                risk_var_95 REAL,
                risk_var_99 REAL,
                risk_cvar_95 REAL,
                risk_cvar_99 REAL,
                risk_skewness REAL,
                risk_kurtosis REAL,
                PRIMARY KEY (ticker, timestamp)
            )
            """)
            
            self.conn.commit()
            
            # Create indices after tables are created and committed
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_ticker ON historical_data(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_timestamp ON historical_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mom_ticker ON momentum_metrics(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mom_timestamp ON momentum_metrics(timestamp)")
            
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def store_historical_data(self, ticker: str, data: pd.DataFrame) -> bool:
        """Store historical price data for a ticker."""
        try:
            # Prepare data for insertion
            df = data.copy()
            df['ticker'] = ticker
            df.index = df.index.strftime('%Y-%m-%d')
            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Insert data
            df.to_sql('historical_data', self.conn, if_exists='replace', index=False)
            self.conn.commit()
            
            logger.info(f"Stored historical data for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error storing historical data for {ticker}: {str(e)}")
            return False

    def get_historical_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve historical data for a ticker."""
        try:
            query = "SELECT * FROM historical_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp"
            
            # Execute query
            df = pd.read_sql_query(query, self.conn, params=params)
            
            if df.empty:
                logger.warning(f"No historical data found for {ticker}")
                return None
                
            # Set timestamp as index and rename columns
            df.set_index('timestamp', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving historical data for {ticker}: {str(e)}")
            return None

    def store_momentum_metrics(self, momentum_df: pd.DataFrame) -> bool:
        """Store momentum metrics."""
        try:
            # Prepare data for insertion
            df = momentum_df.copy()
            df['timestamp'] = datetime.now().strftime('%Y-%m-%d')
            df.reset_index(inplace=True)
            
            # Map column names to database schema
            column_mapping = {
                'Ticker': 'ticker',
                'Last_Price': 'last_price',
                'Avg_Volume': 'avg_volume',
                'composite_score': 'composite_score',
                'ml_score': 'ml_score',
                'enhanced_score': 'enhanced_score',
                '1m_momentum': 'momentum_1m',
                '3m_momentum': 'momentum_3m',
                '6m_momentum': 'momentum_6m',
                '12m_momentum': 'momentum_12m',
                '1m_momentum_rank': 'momentum_1m_rank',
                '3m_momentum_rank': 'momentum_3m_rank',
                '6m_momentum_rank': 'momentum_6m_rank',
                '12m_momentum_rank': 'momentum_12m_rank'
            }
            
            # Rename columns
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure all required columns exist
            required_columns = ['ticker', 'timestamp', 'last_price', 'avg_volume', 'composite_score',
                              'ml_score', 'enhanced_score', 'momentum_1m', 'momentum_3m', 'momentum_6m',
                              'momentum_12m', 'momentum_1m_rank', 'momentum_3m_rank', 'momentum_6m_rank',
                              'momentum_12m_rank']
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Insert data
            df.to_sql('momentum_metrics', self.conn, if_exists='append', index=False)
            self.conn.commit()
            
            logger.info("Stored momentum metrics successfully")
            return True
        except Exception as e:
            logger.error(f"Error storing momentum metrics: {str(e)}")
            return False

    def get_momentum_metrics(self, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve momentum metrics for a specific date."""
        try:
            if date is None:
                # Get latest date
                cursor = self.conn.cursor()
                cursor.execute("SELECT MAX(timestamp) FROM momentum_metrics")
                date = cursor.fetchone()[0]
            
            query = "SELECT * FROM momentum_metrics WHERE timestamp = ?"
            df = pd.read_sql_query(query, self.conn, params=[date])
            
            if df.empty:
                logger.warning(f"No momentum metrics found for date {date}")
                return None
                
            # Set ticker as index
            df.set_index('ticker', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving momentum metrics: {str(e)}")
            return None

    def get_momentum_history(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve historical momentum metrics for a ticker."""
        try:
            query = "SELECT * FROM momentum_metrics WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp"
            
            # Execute query
            df = pd.read_sql_query(query, self.conn, params=params)
            
            if df.empty:
                logger.warning(f"No momentum history found for {ticker}")
                return None
                
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving momentum history for {ticker}: {str(e)}")
            return None

    def __del__(self):
        """Close database connection."""
        try:
            self.conn.close()
        except:
            pass 