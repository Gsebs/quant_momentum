// Initialize Chart.js with default config
let performanceChart;
const chartConfig = {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Portfolio Value',
            data: [],
            borderColor: '#007bff',
            backgroundColor: 'rgba(0, 123, 255, 0.1)',
            borderWidth: 2,
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
                intersect: false
            }
        },
        scales: {
            x: {
                grid: {
                    display: false
                }
            },
            y: {
                beginAtZero: false,
                grid: {
                    color: 'rgba(0, 0, 0, 0.05)'
                }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        },
        animation: {
            duration: 750,
            easing: 'easeInOutQuart'
        }
    }
};

// Initialize dashboard on load
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
});

function initializeDashboard() {
    try {
        // Initialize performance chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        performanceChart = new Chart(ctx, chartConfig);
        
        // Start periodic updates
        updateDashboard();
        setInterval(updateDashboard, 5000); // Update every 5 seconds
        
        // Initialize chart controls
        document.querySelectorAll('.chart-controls button').forEach(button => {
            button.addEventListener('click', (e) => {
                document.querySelectorAll('.chart-controls button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                // TODO: Implement time range filtering
            });
        });
    } catch (error) {
        showErrorMessage('Failed to initialize dashboard: ' + error.message);
    }
}

async function updateDashboard() {
    try {
        // Fetch momentum signals
        const signalsResponse = await fetch('/api/momentum-signals');
        const signalsData = await signalsResponse.json();
        
        if (signalsData.status === 'error') {
            throw new Error(signalsData.message);
        }
        
        // Fetch performance data
        const performanceResponse = await fetch('/api/performance');
        const performanceData = await performanceResponse.json();
        
        if (performanceData.status === 'error') {
            throw new Error(performanceData.message);
        }
        
        // Update UI components
        updateSignalsTable(signalsData.data);
        updatePerformanceMetrics(performanceData.data);
        updatePortfolioChart(performanceData.data);
        updateTradesTable(performanceData.data.recent_trades);
        updatePositionsTable(performanceData.data.positions);
        
        // Update last update time
        document.querySelector('.last-update').textContent = 
            `Last updated: ${new Date().toLocaleTimeString()}`;
            
        // Clear any error messages
        clearAlertMessages();
        
    } catch (error) {
        showErrorMessage('Failed to update dashboard: ' + error.message);
    }
}

function updateSignalsTable(signals) {
    const tbody = document.getElementById('signalsTable');
    tbody.innerHTML = '';
    
    signals.forEach(signal => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${signal.ticker}</td>
            <td class="value-${signal.momentum_score > 0 ? 'positive' : 'negative'}">
                ${signal.momentum_score.toFixed(2)}
            </td>
            <td class="signal-${signal.signal.toLowerCase()}">${signal.signal}</td>
            <td>$${signal.current_price.toFixed(2)}</td>
            <td class="value-${signal.change.startsWith('+') ? 'positive' : 'negative'}">
                ${signal.change}
            </td>
        `;
        tbody.appendChild(row);
    });
}

function updatePerformanceMetrics(data) {
    // Update portfolio value
    const portfolioValue = document.querySelector('.portfolio-value');
    portfolioValue.textContent = `$${data.portfolio_value.toLocaleString()}`;
    
    // Update portfolio change
    const portfolioChange = document.querySelector('.portfolio-change');
    const dailyReturn = data.daily_return * 100;
    portfolioChange.textContent = `${dailyReturn >= 0 ? '+' : ''}${dailyReturn.toFixed(2)}%`;
    portfolioChange.className = `text-muted portfolio-change value-${dailyReturn >= 0 ? 'positive' : 'negative'}`;
    
    // Update cash balance
    document.querySelector('.cash-balance').textContent = 
        `$${data.cash.toLocaleString()}`;
    
    // Update daily return
    document.querySelector('.daily-return').textContent = 
        `${dailyReturn >= 0 ? '+' : ''}${dailyReturn.toFixed(2)}%`;
    
    // Update win rate
    document.querySelector('.win-rate').textContent = 
        `${data.win_rate.toFixed(1)}%`;
}

function updatePortfolioChart(data) {
    // Update chart data
    const returns = data.daily_returns || [];
    const dates = returns.map((_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - (returns.length - i - 1));
        return date.toLocaleDateString();
    });
    
    performanceChart.data.labels = dates;
    performanceChart.data.datasets[0].data = returns.map(r => (1 + r).toFixed(4));
    performanceChart.update();
}

function updateTradesTable(trades) {
    const tbody = document.getElementById('tradesTable');
    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(trade.time).toLocaleString()}</td>
            <td>${trade.ticker}</td>
            <td class="signal-${trade.type.toLowerCase()}">${trade.type}</td>
            <td>$${trade.price.toFixed(2)}</td>
            <td>${trade.quantity}</td>
            <td>$${trade.total.toFixed(2)}</td>
            <td>
                <span class="badge bg-${trade.status === 'FILLED' ? 'success' : 'warning'}">
                    ${trade.status}
                </span>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function updatePositionsTable(positions) {
    const tbody = document.getElementById('positionsTable');
    tbody.innerHTML = '';
    
    Object.entries(positions).forEach(([ticker, position]) => {
        const pnl = position.unrealized_pnl;
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${ticker}</td>
            <td>${position.quantity}</td>
            <td>$${position.price.toFixed(2)}</td>
            <td>$${(position.market_value / position.quantity).toFixed(2)}</td>
            <td class="value-${pnl >= 0 ? 'positive' : 'negative'}">
                ${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}
            </td>
        `;
        tbody.appendChild(row);
    });
}

function showErrorMessage(message) {
    const alertContainer = document.getElementById('alertMessages');
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    alertContainer.appendChild(alert);
}

function clearAlertMessages() {
    const alertContainer = document.getElementById('alertMessages');
    alertContainer.innerHTML = '';
} 