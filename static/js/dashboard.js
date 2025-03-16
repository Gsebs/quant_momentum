// Initialize performance chart
let performanceChart;
let lastUpdateTime = new Date();
let isLoading = false;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    updateDashboard();
    // Update every 30 seconds
    setInterval(updateDashboard, 30000);
    
    // Add tooltips to metrics
    addTooltips();
});

// Add tooltips to explain metrics
function addTooltips() {
    const tooltips = {
        'portfolioValue': 'Current total value of all positions',
        'dailyReturn': 'Percentage return for the current trading day',
        'sharpeRatio': 'Risk-adjusted return metric (higher is better)',
        'maxDrawdown': 'Largest peak-to-trough decline',
        'winRate': 'Percentage of profitable trades',
        'profitFactor': 'Ratio of winning trades to losing trades',
        'modelAccuracy': 'Percentage of correct predictions',
        'signalStrength': 'Average strength of current momentum signals'
    };
    
    Object.entries(tooltips).forEach(([id, text]) => {
        const element = document.getElementById(id);
        if (element) {
            element.setAttribute('title', text);
            element.style.cursor = 'help';
        }
    });
}

// Initialize the performance chart with improved styling
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
                tension: 0.4,
                borderWidth: 2
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
                            return `Portfolio Value: $${context.parsed.y.toLocaleString('en-US', {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2
                            })}`;
                        }
                    },
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
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
                            return '$' + value.toLocaleString('en-US', {
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0
                            });
                        },
                        font: {
                            size: 12
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 12
                        }
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
    if (isLoading) return;
    
    try {
        isLoading = true;
        showLoadingState(true);
        
        // Update momentum signals
        const signalsResponse = await fetch('/api/signals');
        const signalsData = await signalsResponse.json();
        
        // Update performance data
        const performanceResponse = await fetch('/api/performance');
        const performanceData = await performanceResponse.json();
        
        if (performanceData.status === 'success') {
            // Update all metrics
            updatePortfolioStats(performanceData.portfolio_stats);
            updateStrategyPerformance(performanceData.strategy_performance);
            updateModelMetrics(performanceData.model_metrics);
            updateSignalsTable(signalsData);
            updatePerformanceChart(performanceData.performance_data);
            updateRecentTrades(performanceData.recent_trades);
            
            // Flash updated values
            flashUpdatedValues();
            
            // Update last update time
            lastUpdateTime = new Date();
            document.getElementById('last-update').textContent = 
                `Last updated: ${lastUpdateTime.toLocaleTimeString()}`;
                
            showError(null);
        } else {
            showError('Failed to update dashboard data');
        }
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showError('Failed to connect to server');
    } finally {
        isLoading = false;
        showLoadingState(false);
    }
}

// Show/hide loading state
function showLoadingState(show) {
    const elements = document.querySelectorAll('.loading-indicator');
    elements.forEach(el => {
        el.style.display = show ? 'block' : 'none';
    });
    
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        table.style.opacity = show ? '0.6' : '1';
    });
}

// Show/hide error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        if (message) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        } else {
            errorDiv.style.display = 'none';
        }
    }
}

// Update the signals table with improved formatting and signal strength visualization
function updateSignalsTable(signals) {
    const tbody = document.getElementById('signalsTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    if (!Array.isArray(signals)) {
        signals = Object.entries(signals).map(([ticker, signal]) => ({
            ticker,
            ...signal
        }));
    }
    
    signals.forEach((signal) => {
        const row = tbody.insertRow();
        row.className = 'signal-row';
        
        // Ticker
        const tickerCell = row.insertCell(0);
        tickerCell.textContent = signal.ticker;
        tickerCell.className = 'fw-bold';
        
        // Signal with strength visualization
        const signalCell = row.insertCell(1);
        const momentum_score = parseFloat(signal.momentum_score);
        const signalType = momentum_score > 0.1 ? 'BUY' : momentum_score < -0.1 ? 'SELL' : 'HOLD';
        
        // Create signal strength bar
        const strengthBar = document.createElement('div');
        strengthBar.className = 'signal-strength-bar';
        strengthBar.style.width = '60px';
        strengthBar.style.height = '8px';
        strengthBar.style.backgroundColor = '#e9ecef';
        strengthBar.style.borderRadius = '4px';
        strengthBar.style.overflow = 'hidden';
        strengthBar.style.display = 'inline-block';
        strengthBar.style.marginRight = '8px';
        
        const strengthFill = document.createElement('div');
        strengthFill.style.width = `${Math.abs(momentum_score * 100)}%`;
        strengthFill.style.height = '100%';
        strengthFill.style.backgroundColor = signalType === 'BUY' ? '#28a745' : 
                                           signalType === 'SELL' ? '#dc3545' : '#ffc107';
        strengthBar.appendChild(strengthFill);
        
        signalCell.appendChild(strengthBar);
        const signalText = document.createElement('span');
        signalText.textContent = signalType;
        signalText.className = `${signalType === 'BUY' ? 'positive' : signalType === 'SELL' ? 'negative' : 'neutral'} fw-bold`;
        signalCell.appendChild(signalText);
        
        // Score
        const scoreCell = row.insertCell(2);
        scoreCell.textContent = momentum_score.toFixed(2);
        scoreCell.className = `${momentum_score > 0 ? 'positive' : momentum_score < 0 ? 'negative' : 'neutral'} fw-bold`;
        
        // Price
        const priceCell = row.insertCell(3);
        priceCell.textContent = `$${parseFloat(signal.current_price).toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
        
        // Change
        const changeCell = row.insertCell(4);
        const change = parseFloat(signal.price_change) * 100;
        changeCell.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
        changeCell.className = `${change >= 0 ? 'positive' : 'negative'} fw-bold`;
    });
}

// Update portfolio statistics with improved animations
function updatePortfolioStats(stats) {
    if (!stats) return;
    
    // Portfolio value with currency formatting
    animateValue('portfolioValue', stats.portfolio_value, '$', '', true);
    
    // Daily return with color coding
    const dailyReturnElement = document.getElementById('dailyReturn');
    const dailyReturn = stats.daily_return;
    dailyReturnElement.textContent = `${dailyReturn >= 0 ? '+' : ''}${dailyReturn.toFixed(2)}%`;
    dailyReturnElement.className = `badge ${dailyReturn >= 0 ? 'bg-success' : 'bg-danger'}`;
    
    // Risk metrics
    animateValue('sharpeRatio', stats.sharpe_ratio, '', '', false);
    animateValue('maxDrawdown', stats.max_drawdown, '', '%', false);
}

// Update strategy performance metrics
function updateStrategyPerformance(performance) {
    if (!performance) return;
    
    animateValue('winRate', performance.win_rate, '', '%', false);
    animateValue('profitFactor', performance.profit_factor, '', '', false);
}

// Update model metrics
function updateModelMetrics(metrics) {
    if (!metrics) return;
    
    animateValue('modelAccuracy', metrics.prediction_accuracy, '', '%', false);
    animateValue('signalStrength', metrics.signal_strength, '', '', false);
}

// Animate value changes with improved smoothness
function animateValue(elementId, newValue, prefix = '', suffix = '', isCurrency = false) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const oldValue = parseFloat(element.textContent.replace(/[^0-9.-]+/g, ''));
    const duration = 1000;
    const steps = 60;
    const increment = (newValue - oldValue) / steps;
    
    let currentStep = 0;
    const interval = setInterval(() => {
        currentStep++;
        const currentValue = oldValue + (increment * currentStep);
        
        element.textContent = `${prefix}${isCurrency ? 
            currentValue.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }) :
            currentValue.toFixed(2)}${suffix}`;
        
        if (currentStep >= steps) {
            clearInterval(interval);
            element.textContent = `${prefix}${isCurrency ? 
                newValue.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }) :
                newValue.toFixed(2)}${suffix}`;
        }
    }, duration / steps);
}

// Update performance chart with improved animations
function updatePerformanceChart(data) {
    if (!data || !data.dates || !data.values) return;
    
    performanceChart.data.labels = data.dates;
    performanceChart.data.datasets[0].data = data.values;
    
    // Update chart with smooth animation
    performanceChart.update('active');
}

// Update recent trades table with improved formatting
function updateRecentTrades(trades) {
    if (!trades || !trades.length) return;
    
    const tbody = document.getElementById('tradesTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        const row = tbody.insertRow();
        row.className = 'trade-row';
        
        // Time
        const timeCell = row.insertCell(0);
        const time = new Date(trade.time);
        timeCell.textContent = time.toLocaleString();
        
        // Ticker
        const tickerCell = row.insertCell(1);
        tickerCell.textContent = trade.ticker;
        tickerCell.className = 'fw-bold';
        
        // Type
        const typeCell = row.insertCell(2);
        typeCell.textContent = trade.type;
        typeCell.className = `${trade.type === 'BUY' ? 'positive' : 'negative'} fw-bold`;
        
        // Price
        const priceCell = row.insertCell(3);
        priceCell.textContent = `$${trade.price.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
        
        // Quantity
        const quantityCell = row.insertCell(4);
        quantityCell.textContent = trade.quantity.toLocaleString('en-US');
    });
}

// Flash updated values with improved animation
function flashUpdatedValues() {
    const elements = document.querySelectorAll('.signal-row, .trade-row');
    elements.forEach(element => {
        element.classList.add('flash');
        setTimeout(() => element.classList.remove('flash'), 500);
    });
} 