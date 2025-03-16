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
                intersect: false,
                callbacks: {
                    label: function(context) {
                        return `$${parseFloat(context.raw).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                    }
                }
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
                },
                ticks: {
                    callback: function(value) {
                        return '$' + value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
                    }
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
        if (!signalsResponse.ok) {
            throw new Error(`HTTP error! status: ${signalsResponse.status}`);
        }
        const signalsData = await signalsResponse.json();
        
        if (signalsData.status === 'error') {
            throw new Error(signalsData.message);
        }
        
        // Fetch performance data
        const performanceResponse = await fetch('/api/performance');
        if (!performanceResponse.ok) {
            throw new Error(`HTTP error! status: ${performanceResponse.status}`);
        }
        const performanceData = await performanceResponse.json();
        
        if (performanceData.status === 'error') {
            throw new Error(performanceData.message);
        }
        
        // Update UI components
        if (signalsData.data) {
            updateSignalsTable(signalsData.data);
        }
        
        if (performanceData.data) {
            updatePerformanceMetrics(performanceData.data);
            updatePortfolioChart(performanceData.data);
            updateTradesTable(performanceData.data.trades || []);
            updatePositionsTable(performanceData.data.positions || {});
        }
        
        // Update last update time
        const lastUpdate = document.querySelector('.last-update');
        if (lastUpdate) {
            lastUpdate.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
        }
            
        // Clear any error messages
        clearAlertMessages();
        
    } catch (error) {
        showErrorMessage('Failed to update dashboard: ' + error.message);
    }
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

function updateSignalsTable(signals) {
    const tbody = document.getElementById('signalsTable');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    signals.forEach(signal => {
        const row = document.createElement('tr');
        const score = parseFloat(signal.momentum_score);
        row.innerHTML = `
            <td>${signal.ticker}</td>
            <td class="value-${score > 0 ? 'positive' : 'negative'}">
                ${score.toFixed(2)}
            </td>
            <td class="signal-${signal.signal.toLowerCase()}">${signal.signal}</td>
            <td>${formatCurrency(signal.current_price)}</td>
            <td class="value-${signal.change.startsWith('+') ? 'positive' : 'negative'}">
                ${signal.change}
            </td>
        `;
        tbody.appendChild(row);
    });
}

function updatePerformanceMetrics(data) {
    try {
        // Update portfolio value
        const portfolioValue = document.querySelector('.portfolio-value');
        if (portfolioValue) {
            portfolioValue.textContent = formatCurrency(data.portfolio_value || 0);
        }
        
        // Update portfolio change
        const portfolioChange = document.querySelector('.portfolio-change');
        if (portfolioChange) {
            const dailyReturn = (data.daily_return || 0) * 100;
            portfolioChange.textContent = `${dailyReturn >= 0 ? '+' : ''}${dailyReturn.toFixed(2)}%`;
            portfolioChange.className = `text-muted portfolio-change value-${dailyReturn >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update cash balance
        const cashBalance = document.querySelector('.cash-balance');
        if (cashBalance) {
            cashBalance.textContent = formatCurrency(data.cash || 0);
        }
        
        // Update daily return
        const dailyReturnElement = document.querySelector('.daily-return');
        if (dailyReturnElement) {
            const dailyReturn = (data.daily_return || 0) * 100;
            dailyReturnElement.textContent = `${dailyReturn >= 0 ? '+' : ''}${dailyReturn.toFixed(2)}%`;
            dailyReturnElement.className = `daily-return value-${dailyReturn >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update win rate
        const winRate = document.querySelector('.win-rate');
        if (winRate) {
            winRate.textContent = `${(data.win_rate || 0).toFixed(1)}%`;
        }
    } catch (error) {
        console.error('Error updating performance metrics:', error);
    }
}

function updatePortfolioChart(data) {
    try {
        if (!data.portfolio_history || !Array.isArray(data.portfolio_history)) {
            console.warn('No portfolio history data available');
            return;
        }
        
        const history = data.portfolio_history;
        const dates = history.map(item => new Date(item.timestamp).toLocaleDateString());
        const values = history.map(item => item.value);
        
        performanceChart.data.labels = dates;
        performanceChart.data.datasets[0].data = values;
        performanceChart.update();
    } catch (error) {
        console.error('Error updating portfolio chart:', error);
    }
}

function updateTradesTable(trades) {
    const tbody = document.getElementById('tradesTable');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        try {
            const row = document.createElement('tr');
            const tradeTime = new Date(trade.time);
            
            if (isNaN(tradeTime.getTime())) {
                console.warn('Invalid trade time:', trade.time);
                return;
            }
            
            row.innerHTML = `
                <td>${tradeTime.toLocaleString()}</td>
                <td>${trade.ticker}</td>
                <td class="signal-${trade.type.toLowerCase()}">${trade.type}</td>
                <td>${formatCurrency(trade.price)}</td>
                <td>${trade.quantity}</td>
                <td>${formatCurrency(trade.total)}</td>
                <td>
                    <span class="badge bg-${trade.status === 'FILLED' ? 'success' : 'warning'}">
                        ${trade.status}
                    </span>
                </td>
            `;
            tbody.appendChild(row);
        } catch (error) {
            console.error('Error adding trade row:', error);
        }
    });
}

function updatePositionsTable(positions) {
    const tbody = document.getElementById('positionsTable');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    Object.entries(positions).forEach(([ticker, position]) => {
        try {
            const pnl = position.unrealized_pnl || 0;
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${ticker}</td>
                <td>${position.quantity}</td>
                <td>${formatCurrency(position.price)}</td>
                <td>${formatCurrency(position.market_value / position.quantity)}</td>
                <td class="value-${pnl >= 0 ? 'positive' : 'negative'}">
                    ${pnl >= 0 ? '+' : ''}${formatCurrency(Math.abs(pnl))}
                </td>
            `;
            tbody.appendChild(row);
        } catch (error) {
            console.error('Error adding position row:', error);
        }
    });
}

function showErrorMessage(message) {
    const alertContainer = document.getElementById('alertMessages');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    alertContainer.appendChild(alert);
    
    // Auto-remove alert after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

function clearAlertMessages() {
    const alertContainer = document.getElementById('alertMessages');
    if (alertContainer) {
        alertContainer.innerHTML = '';
    }
} 