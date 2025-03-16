// Initialize performance chart
let performanceChart;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    updateDashboard();
    // Update every 60 seconds
    setInterval(updateDashboard, 60000);
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
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
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
        updateSignalsTable(signalsData.data);
        
        // Update performance data
        const performanceResponse = await fetch('/api/performance');
        const performanceData = await performanceResponse.json();
        updatePerformanceData(performanceData.performance);
        
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
    
    for (const ticker in signals) {
        const signal = signals[ticker];
        const row = tbody.insertRow();
        
        // Add cells
        row.insertCell(0).textContent = ticker;
        row.insertCell(1).textContent = signal.signal;
        row.insertCell(2).textContent = signal.score.toFixed(2);
        row.insertCell(3).textContent = `$${signal.current_price.toFixed(2)}`;
        
        const changeCell = row.insertCell(4);
        const change = signal.price_change;
        changeCell.textContent = `${(change * 100).toFixed(2)}%`;
        changeCell.className = change >= 0 ? 'positive' : 'negative';
    }
}

// Update performance data and chart
function updatePerformanceData(performance) {
    if (!performance || performance.length === 0) return;
    
    // Update portfolio statistics
    const latest = performance[performance.length - 1];
    document.getElementById('portfolioValue').textContent = 
        `$${latest.portfolio_value.toFixed(2)}`;
    document.getElementById('dailyReturn').textContent = 
        `${(latest.daily_return * 100).toFixed(2)}%`;
    document.getElementById('sharpeRatio').textContent = 
        latest.sharpe_ratio.toFixed(2);
    document.getElementById('maxDrawdown').textContent = 
        `${(latest.max_drawdown * 100).toFixed(2)}%`;
    
    // Update chart
    const dates = performance.map(p => new Date(p.date).toLocaleDateString());
    const values = performance.map(p => p.portfolio_value);
    
    performanceChart.data.labels = dates;
    performanceChart.data.datasets[0].data = values;
    performanceChart.update();
    
    // Update recent trades
    updateRecentTrades(performance);
}

// Update recent trades table
function updateRecentTrades(performance) {
    const tbody = document.getElementById('tradesTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    // Get the last 10 trades
    const trades = performance.slice(-10).reverse();
    
    trades.forEach(trade => {
        if (trade.trades && trade.trades.length > 0) {
            trade.trades.forEach(t => {
                const row = tbody.insertRow();
                row.insertCell(0).textContent = new Date(trade.date).toLocaleString();
                row.insertCell(1).textContent = t.ticker;
                row.insertCell(2).textContent = t.type;
                row.insertCell(3).textContent = `$${t.price.toFixed(2)}`;
                row.insertCell(4).textContent = t.quantity;
            });
        }
    });
} 