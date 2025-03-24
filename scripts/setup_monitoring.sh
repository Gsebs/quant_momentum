#!/bin/bash

# Create necessary directories
mkdir -p prometheus/data
mkdir -p grafana/data
mkdir -p logs

# Set up Prometheus
cat > prometheus.yml << EOL
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOL

# Set up Grafana dashboards
cat > grafana/dashboards/trading.json << EOL
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          },
          "expr": "rate(trading_trades_total[5m])",
          "interval": "",
          "legendFormat": "Trades per second",
          "refId": "A"
        }
      ],
      "title": "Trading Volume",
      "type": "timeseries"
    }
  ],
  "refresh": "5s",
  "schemaVersion": 38,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Trading Dashboard",
  "version": 0,
  "weekStart": ""
}
EOL

# Set up log rotation
cat > /etc/logrotate.d/trading-app << EOL
/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload prometheus
    endscript
}
EOL

# Set up monitoring alerts
cat > prometheus/alerts.yml << EOL
groups:
  - name: trading_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(trading_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
          description: "Error rate is above 0.1 per second for 5 minutes"

      - alert: HighLatency
        expr: trading_latency_seconds > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High latency detected
          description: "Trading latency is above 1 second for 5 minutes"

      - alert: DatabaseConnection
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Database connection lost
          description: "Cannot connect to the database"
EOL

# Set up Grafana datasource
cat > grafana/provisioning/datasources/prometheus.yml << EOL
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOL

# Set up Grafana dashboards provisioning
cat > grafana/provisioning/dashboards/trading.yml << EOL
apiVersion: 1

providers:
  - name: 'Trading'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
EOL

# Set permissions
chmod -R 777 prometheus/data
chmod -R 777 grafana/data
chmod -R 777 logs

echo "Monitoring setup completed successfully!" 