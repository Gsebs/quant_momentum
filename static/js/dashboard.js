// Dashboard initialization and real-time updates
let performanceChart;
let lastUpdateTime = new Date();
const updateInterval = 5000; // Update every 5 seconds

// Initialize the dashboard
async function initializeDashboard() {
    // Initialize performance chart
    const ctx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `$${context.parsed.y.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
                        }
                    }
                },
                legend: {
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                }
            }
        }
    });

    // Start real-time updates
    updateDashboard();
    setInterval(updateDashboard, updateInterval);
}

// Update dashboard with latest data
async function updateDashboard() {
    try {
        // Fetch momentum signals and performance data
        const [signalsResponse, performanceResponse] = await Promise.all([
            fetch('/api/momentum-signals'),
            fetch('/api/performance')
        ]);

        const signalsData = await signalsResponse.json();
        const performanceData = await performanceResponse.json();

        if (signalsData.status === 'success' && performanceData.status === 'success') {
            updateSignalsTable(signalsData.data);
            updatePerformanceMetrics(performanceData.data);
            updatePortfolioChart(performanceData.data);
            updateTradesTable(performanceData.data.recent_trades);
            updateLastUpdateTime();
        }
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showErrorMessage('Failed to update dashboard. Retrying...');
    }
}

// Update signals table with latest momentum data
function updateSignalsTable(signals) {
    const table = document.getElementById('signalsTable');
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    signals.forEach(signal => {
        const row = document.createElement('tr');
        const momentumClass = signal.momentum_score > 0.1 ? 'positive' : 
                            signal.momentum_score < -0.1 ? 'negative' : 'neutral';
        
        row.innerHTML = `
            <td class="font-weight-bold">${signal.ticker}</td>
            <td class="${momentumClass}">${signal.momentum_score.toFixed(3)}</td>
            <td>$${parseFloat(signal.current_price).toFixed(2)}</td>
            <td class="signal-type ${signal.signal.toLowerCase()}">${signal.signal}</td>
        `;
        tbody.appendChild(row);
    });
}

// Update performance metrics with animations
function updatePerformanceMetrics(data) {
    // Portfolio value with animation
    animateValue('portfolioValue', data.portfolio_value, '$');
    animateValue('cashBalance', data.cash, '$');
    
    // Performance metrics
    document.getElementById('dailyReturn').textContent = 
        `${(data.daily_return * 100).toFixed(2)}%`;
    document.getElementById('sharpeRatio').textContent = 
        data.sharpe_ratio.toFixed(2);
    document.getElementById('winRate').textContent = 
        `${data.win_rate.toFixed(1)}%`;
    
    // Update positions table
    updatePositionsTable(data.positions);
}

// Animate value changes
function animateValue(elementId, newValue, prefix = '') {
    const element = document.getElementById(elementId);
    const startValue = parseFloat(element.getAttribute('data-value') || '0');
    const duration = 1000;
    const steps = 60;
    const increment = (newValue - startValue) / steps;
    
    let currentStep = 0;
    const interval = setInterval(() => {
        currentStep++;
        const currentValue = startValue + (increment * currentStep);
        element.textContent = `${prefix}${currentValue.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
        
        if (currentStep >= steps) {
            clearInterval(interval);
            element.setAttribute('data-value', newValue);
        }
    }, duration / steps);
}

// Update portfolio chart
function updatePortfolioChart(data) {
    const timestamps = data.daily_returns.map((_, index) => {
        const date = new Date();
        date.setDate(date.getDate() - (data.daily_returns.length - index - 1));
        return date.toLocaleDateString();
    });

    const portfolioValues = data.daily_returns.map((return_value, index) => {
        return data.portfolio_value * (1 + return_value);
    });

    performanceChart.data.labels = timestamps;
    performanceChart.data.datasets[0].data = portfolioValues;
    performanceChart.update();
}

// Update trades table with latest trades
function updateTradesTable(trades) {
    const table = document.getElementById('tradesTable');
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    trades.reverse().forEach(trade => {
        const row = document.createElement('tr');
        const typeClass = trade.type === 'BUY' ? 'buy' : 'sell';
        
        row.innerHTML = `
            <td>${trade.time}</td>
            <td class="font-weight-bold">${trade.ticker}</td>
            <td class="${typeClass}">${trade.type}</td>
            <td>$${trade.price.toFixed(2)}</td>
            <td>${trade.quantity}</td>
            <td>$${(trade.price * trade.quantity).toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
}

// Update positions table
function updatePositionsTable(positions) {
    const table = document.getElementById('positionsTable');
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    Object.entries(positions).forEach(([ticker, position]) => {
        const row = document.createElement('tr');
        const value = position.quantity * position.price;
        
        row.innerHTML = `
            <td class="font-weight-bold">${ticker}</td>
            <td>${position.quantity}</td>
            <td>$${position.price.toFixed(2)}</td>
            <td>$${value.toFixed(2)}</td>
            <td>${new Date(position.timestamp).toLocaleString()}</td>
        `;
        tbody.appendChild(row);
    });
}

// Update last update time
function updateLastUpdateTime() {
    lastUpdateTime = new Date();
    document.getElementById('lastUpdate').textContent = 
        `Last updated: ${lastUpdateTime.toLocaleString()}`;
}

// Show error message
function showErrorMessage(message) {
    const alertDiv = document.getElementById('alertMessage');
    alertDiv.textContent = message;
    alertDiv.classList.remove('d-none');
    setTimeout(() => alertDiv.classList.add('d-none'), 5000);
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeDashboard); 