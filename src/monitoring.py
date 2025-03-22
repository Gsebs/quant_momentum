"""
Performance Monitoring for Quantitative HFT Algorithm.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import psutil
import redis
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LatencyMetric:
    """Represents a latency measurement."""
    timestamp: datetime
    operation: str
    duration_ms: float

@dataclass
class SystemMetric:
    """Represents system resource usage."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]

class PerformanceMonitor:
    """Monitors and tracks system performance metrics."""
    
    def __init__(self, config: Dict):
        """Initialize the performance monitor with configuration."""
        self.config = config
        self.metrics_window = config.get('metrics_window', 1000)
        self.latency_threshold_ms = config.get('latency_threshold_ms', 1.0)
        self.alert_threshold = config.get('alert_threshold', 0.95)
        
        # Initialize metric storage
        self.latency_metrics: deque = deque(maxlen=self.metrics_window)
        self.system_metrics: deque = deque(maxlen=self.metrics_window)
        self.trading_metrics: Dict[str, deque] = {
            'pnl': deque(maxlen=self.metrics_window),
            'positions': deque(maxlen=self.metrics_window),
            'trades': deque(maxlen=self.metrics_window)
        }
        
        # Initialize Redis connection for real-time metrics
        self.redis_client = redis.Redis.from_url(
            config.get('redis_url', 'redis://localhost:6379/0')
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def record_latency(self, operation: str, duration_ms: float) -> None:
        """Record latency for an operation."""
        try:
            metric = LatencyMetric(
                timestamp=datetime.now(),
                operation=operation,
                duration_ms=duration_ms
            )
            
            self.latency_metrics.append(metric)
            
            # Store in Redis for real-time monitoring
            self.redis_client.hset(
                'latency_metrics',
                operation,
                f"{duration_ms:.3f}"
            )
            
            # Check for latency threshold violation
            if duration_ms > self.latency_threshold_ms:
                self.logger.warning(
                    f"High latency detected for {operation}: {duration_ms:.3f}ms"
                )
                
        except Exception as e:
            self.logger.error(f"Error recording latency metric: {str(e)}")
            
    def record_system_metrics(self) -> None:
        """Record system resource usage metrics."""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()._asdict()
            
            # Get network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            metric = SystemMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_io=disk_io,
                network_io=network_io
            )
            
            self.system_metrics.append(metric)
            
            # Store in Redis for real-time monitoring
            self.redis_client.hmset('system_metrics', {
                'cpu_percent': f"{cpu_percent:.1f}",
                'memory_percent': f"{memory_percent:.1f}",
                'disk_read_bytes': str(disk_io['read_bytes']),
                'disk_write_bytes': str(disk_io['write_bytes']),
                'net_bytes_sent': str(network_io['bytes_sent']),
                'net_bytes_recv': str(network_io['bytes_recv'])
            })
            
            # Check for resource usage alerts
            if cpu_percent > self.alert_threshold * 100:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > self.alert_threshold * 100:
                self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Error recording system metrics: {str(e)}")
            
    def record_trading_metrics(self, metrics: Dict) -> None:
        """Record trading performance metrics."""
        try:
            timestamp = datetime.now()
            
            # Update metric queues
            for metric_type, value in metrics.items():
                if metric_type in self.trading_metrics:
                    self.trading_metrics[metric_type].append({
                        'timestamp': timestamp,
                        'value': value
                    })
                    
            # Store in Redis for real-time monitoring
            self.redis_client.hmset('trading_metrics', {
                k: str(v) for k, v in metrics.items()
            })
            
        except Exception as e:
            self.logger.error(f"Error recording trading metrics: {str(e)}")
            
    def get_latency_stats(self, operation: Optional[str] = None) -> Dict:
        """Get latency statistics for operations."""
        try:
            metrics = [m for m in self.latency_metrics if not operation or m.operation == operation]
            if not metrics:
                return {}
                
            durations = [m.duration_ms for m in metrics]
            return {
                'mean': np.mean(durations),
                'median': np.median(durations),
                'p95': np.percentile(durations, 95),
                'p99': np.percentile(durations, 99),
                'max': np.max(durations),
                'min': np.min(durations)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating latency stats: {str(e)}")
            return {}
            
    def get_system_stats(self, window_minutes: int = 5) -> Dict:
        """Get system resource usage statistics."""
        try:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            metrics = [m for m in self.system_metrics if m.timestamp >= cutoff]
            if not metrics:
                return {}
                
            return {
                'cpu': {
                    'mean': np.mean([m.cpu_percent for m in metrics]),
                    'max': np.max([m.cpu_percent for m in metrics])
                },
                'memory': {
                    'mean': np.mean([m.memory_percent for m in metrics]),
                    'max': np.max([m.memory_percent for m in metrics])
                },
                'disk_io': {
                    'read_bytes': metrics[-1].disk_io['read_bytes'] - metrics[0].disk_io['read_bytes'],
                    'write_bytes': metrics[-1].disk_io['write_bytes'] - metrics[0].disk_io['write_bytes']
                },
                'network_io': {
                    'bytes_sent': metrics[-1].network_io['bytes_sent'] - metrics[0].network_io['bytes_sent'],
                    'bytes_recv': metrics[-1].network_io['bytes_recv'] - metrics[0].network_io['bytes_recv']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating system stats: {str(e)}")
            return {}
            
    def get_trading_stats(self, metric_type: str) -> Dict:
        """Get trading performance statistics."""
        try:
            if metric_type not in self.trading_metrics:
                return {}
                
            metrics = self.trading_metrics[metric_type]
            if not metrics:
                return {}
                
            values = [m['value'] for m in metrics]
            return {
                'current': values[-1],
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trading stats: {str(e)}")
            return {}
            
    def check_health(self) -> Dict[str, str]:
        """Check overall system health."""
        try:
            health_status = {
                'status': 'healthy',
                'latency': 'normal',
                'resources': 'normal',
                'trading': 'normal'
            }
            
            # Check latency health
            latency_stats = self.get_latency_stats()
            if latency_stats.get('p95', 0) > self.latency_threshold_ms:
                health_status['latency'] = 'degraded'
                health_status['status'] = 'degraded'
                
            # Check system resource health
            system_stats = self.get_system_stats(window_minutes=1)
            if (system_stats.get('cpu', {}).get('mean', 0) > self.alert_threshold * 100 or
                system_stats.get('memory', {}).get('mean', 0) > self.alert_threshold * 100):
                health_status['resources'] = 'degraded'
                health_status['status'] = 'degraded'
                
            # Check trading health (example: check if PnL is decreasing)
            pnl_stats = self.get_trading_stats('pnl')
            if pnl_stats.get('current', 0) < 0:
                health_status['trading'] = 'warning'
                
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_metrics_snapshot(self) -> Dict:
        """Get a snapshot of all current metrics."""
        try:
            return {
                'latency': self.get_latency_stats(),
                'system': self.get_system_stats(),
                'trading': {
                    metric_type: self.get_trading_stats(metric_type)
                    for metric_type in self.trading_metrics
                },
                'health': self.check_health()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics snapshot: {str(e)}")
            return {} 