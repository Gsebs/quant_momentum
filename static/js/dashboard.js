// Initialize performance chart
let performanceChart;
let lastUpdateTime = new Date();

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
                            return `$${context.parsed.y.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
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
                            return '$' + value.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
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
            // Update portfolio statistics with animation
            updatePortfolioStats(performanceData.portfolio_stats);
            
            // Update performance chart
            updatePerformanceChart(performanceData.performance_data);
            
            // Update recent trades
            updateRecentTrades(performanceData.recent_trades);
            
            // Flash updated values
            flashUpdatedValues();
        }
        
        // Update last update time
        lastUpdateTime = new Date();
        document.getElementById('last-update').textContent = 
            `Last updated: ${lastUpdateTime.toLocaleTimeString()}`;
            
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
        row.className = 'signal-row';
        
        // Add cells
        row.insertCell(0).textContent = ticker;
        
        const signalCell = row.insertCell(1);
        signalCell.textContent = signal.signal || (signal.momentum_score > 0 ? 'BUY' : 'SELL');
        signalCell.className = signal.momentum_score > 0 ? 'positive' : 'negative';
        
        row.insertCell(2).textContent = signal.momentum_score.toFixed(2);
        
        const priceCell = row.insertCell(3);
        priceCell.textContent = `$${signal.current_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const changeCell = row.insertCell(4);
        const change = signal.price_change * 100;
        changeCell.textContent = `${change.toFixed(2)}%`;
        changeCell.className = change >= 0 ? 'positive' : 'negative';
    });
}

// Update portfolio statistics with animation
function updatePortfolioStats(stats) {
    if (!stats) return;
    
    animateValue('portfolioValue', stats.portfolio_value, '$');
    animateValue('dailyReturn', stats.daily_return, '%');
    animateValue('sharpeRatio', stats.sharpe_ratio);
    animateValue('maxDrawdown', stats.max_drawdown, '%');
}

// Animate value changes
function animateValue(elementId, newValue, prefix = '', suffix = '') {
    const element = document.getElementById(elementId);
    const oldValue = parseFloat(element.textContent.replace(/[^0-9.-]+/g, ''));
    const duration = 1000; // Animation duration in ms
    const steps = 60; // Number of steps in animation
    const increment = (newValue - oldValue) / steps;
    
    let currentStep = 0;
    const interval = setInterval(() => {
        currentStep++;
        const currentValue = oldValue + (increment * currentStep);
        element.textContent = `${prefix}${currentValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}${suffix}`;
        
        if (currentStep >= steps) {
            clearInterval(interval);
            element.textContent = `${prefix}${newValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}${suffix}`;
        }
    }, duration / steps);
}

// Update performance chart
function updatePerformanceChart(data) {
    if (!data || !data.dates || !data.values) return;
    
    performanceChart.data.labels = data.dates;
    performanceChart.data.datasets[0].data = data.values;
    
    // Update chart with animation
    performanceChart.update('active');
}

// Update recent trades table
function updateRecentTrades(trades) {
    if (!trades || !trades.length) return;
    
    const tbody = document.getElementById('tradesTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        const row = tbody.insertRow();
        row.className = 'trade-row';
        
        // Format time
        const time = new Date(trade.time);
        row.insertCell(0).textContent = time.toLocaleString();
        
        // Add other cells
        row.insertCell(1).textContent = trade.ticker;
        
        const typeCell = row.insertCell(2);
        typeCell.textContent = trade.type;
        typeCell.className = trade.type === 'BUY' ? 'positive' : 'negative';
        
        const priceCell = row.insertCell(3);
        priceCell.textContent = `$${trade.price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const quantityCell = row.insertCell(4);
        quantityCell.textContent = trade.quantity.toLocaleString('en-US');
    });
}

// Flash updated values
function flashUpdatedValues() {
    const elements = document.querySelectorAll('.signal-row, .trade-row');
    elements.forEach(element => {
        element.classList.add('flash');
        setTimeout(() => element.classList.remove('flash'), 500);
    });
} 