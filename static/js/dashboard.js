// Initialize performance chart
let performanceChart;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    updateDashboard();
    // Update every 30 seconds
    setInterval(updateDashboard, 30000);
});

// Initialize the performance chart
function initializeChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `$${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Update the dashboard with latest data
async function updateDashboard() {
    try {
        // Update momentum signals
        const signalsResponse = await fetch('/api/momentum-signals');
        const signalsData = await signalsResponse.json();
        if (signalsData.data) {
            updateSignalsTable(signalsData.data);
        }
        
        // Update performance data
        const performanceResponse = await fetch('/api/performance');
        const performanceData = await performanceResponse.json();
        
        if (performanceData.status === 'success') {
            // Update portfolio statistics
            updatePortfolioStats(performanceData.portfolio_stats);
            
            // Update performance chart
            updatePerformanceChart(performanceData.data);
            
            // Update recent trades
            updateRecentTrades(performanceData.recent_trades);
        }
        
        // Update last update time
        document.getElementById('last-update').textContent = 
            `Last updated: ${new Date().toLocaleTimeString()}`;
            
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

// Update the signals table
function updateSignalsTable(signals) {
    const tbody = document.getElementById('signalsTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    Object.entries(signals).forEach(([ticker, signal]) => {
        const row = tbody.insertRow();
        
        // Add cells
        row.insertCell(0).textContent = ticker;
        
        const signalCell = row.insertCell(1);
        signalCell.textContent = signal.momentum_score > 0 ? 'BUY' : 'SELL';
        signalCell.className = signal.momentum_score > 0 ? 'positive' : 'negative';
        
        row.insertCell(2).textContent = signal.momentum_score.toFixed(2);
        row.insertCell(3).textContent = `$${signal.current_price.toFixed(2)}`;
        
        const changeCell = row.insertCell(4);
        const change = signal.price_change;
        changeCell.textContent = `${(change * 100).toFixed(2)}%`;
        changeCell.className = change >= 0 ? 'positive' : 'negative';
    });
}

// Update portfolio statistics
function updatePortfolioStats(stats) {
    if (!stats) return;
    
    document.getElementById('portfolioValue').textContent = 
        `$${stats.portfolio_value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    
    const dailyReturn = document.getElementById('dailyReturn');
    dailyReturn.textContent = `${(stats.daily_return * 100).toFixed(2)}%`;
    dailyReturn.className = stats.daily_return >= 0 ? 'positive' : 'negative';
    
    document.getElementById('sharpeRatio').textContent = 
        stats.sharpe_ratio.toFixed(2);
    
    document.getElementById('maxDrawdown').textContent = 
        `${(stats.max_drawdown * 100).toFixed(2)}%`;
}

// Update performance chart
function updatePerformanceChart(data) {
    if (!data || !data.returns) return;
    
    const returns = data.returns;
    const dates = returns.map(r => new Date(r.Date).toLocaleDateString());
    const values = returns.map(r => r['Portfolio Value']);
    
    performanceChart.data.labels = dates;
    performanceChart.data.datasets[0].data = values;
    performanceChart.update();
}

// Update recent trades table
function updateRecentTrades(trades) {
    if (!trades || !trades.length) return;
    
    const tbody = document.getElementById('tradesTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        const row = tbody.insertRow();
        
        // Format date
        const date = new Date(trade.date);
        row.insertCell(0).textContent = date.toLocaleString();
        
        // Add other cells
        row.insertCell(1).textContent = trade.ticker;
        
        const typeCell = row.insertCell(2);
        typeCell.textContent = trade.type;
        typeCell.className = trade.type === 'BUY' ? 'positive' : 'negative';
        
        row.insertCell(3).textContent = `$${trade.price.toFixed(2)}`;
        row.insertCell(4).textContent = trade.quantity;
    });
} 