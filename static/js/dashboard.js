// Dashboard initialization and real-time updates
let performanceChart;
let lastUpdateTime = new Date();
const updateInterval = 5000; // Update every 5 seconds
let retryCount = 0;
const maxRetries = 3;

// Initialize the dashboard
async function initializeDashboard() {
    try {
        // Show loading state
        document.querySelectorAll('.loading-indicator').forEach(el => el.style.display = 'block');
        
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
                    fill: {
                        target: 'origin',
                        above: 'rgba(75, 192, 192, 0.1)'
                    }
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
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
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
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });

        // Add click handlers for chart time range buttons
        document.querySelectorAll('.chart-controls button').forEach(button => {
            button.addEventListener('click', () => {
                document.querySelectorAll('.chart-controls button').forEach(b => b.classList.remove('active'));
                button.classList.add('active');
                updateChartTimeRange(button.dataset.range);
            });
        });

        // Start real-time updates
        await updateDashboard();
        setInterval(updateDashboard, updateInterval);
        
        // Hide loading state
        document.querySelectorAll('.loading-indicator').forEach(el => el.style.display = 'none');
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showErrorMessage('Failed to initialize dashboard. Please refresh the page.');
    }
}

// Update dashboard with latest data
async function updateDashboard() {
    try {
        const [signalsResponse, performanceResponse] = await Promise.all([
            fetchWithRetry('/api/momentum-signals'),
            fetchWithRetry('/api/performance')
        ]);

        if (!signalsResponse.ok || !performanceResponse.ok) {
            throw new Error('API request failed');
        }

        const signalsData = await signalsResponse.json();
        const performanceData = await performanceResponse.json();

        if (signalsData.status === 'success' && performanceData.status === 'success') {
            updateSignalsTable(signalsData.data);
            updatePerformanceMetrics(performanceData.data);
            updatePortfolioChart(performanceData.data);
            updateTradesTable(performanceData.data.recent_trades);
            updatePositionsTable(performanceData.data.positions);
            updateLastUpdateTime();
            retryCount = 0; // Reset retry count on successful update
            
            // Show success status
            document.querySelector('.status-indicator').classList.remove('error');
            document.querySelector('.status-text').textContent = 'Real-time Updates Active';
        } else {
            throw new Error('Invalid API response');
        }
    } catch (error) {
        console.error('Error updating dashboard:', error);
        retryCount++;
        if (retryCount >= maxRetries) {
            showErrorMessage('Failed to update dashboard after multiple attempts. Please refresh the page.');
            document.querySelector('.status-indicator').classList.add('error');
            document.querySelector('.status-text').textContent = 'Updates Failed';
        } else {
            showErrorMessage(`Failed to update dashboard. Retrying... (${retryCount}/${maxRetries})`);
        }
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
            <td class="${momentumClass}">${parseFloat(signal.momentum_score).toFixed(3)}</td>
            <td>$${parseFloat(signal.current_price).toFixed(2)}</td>
            <td class="signal-type ${signal.signal.toLowerCase()}">${signal.signal}</td>
            <td class="${signal.change.startsWith('+') ? 'positive' : 'negative'}">${signal.change}</td>
        `;
        tbody.appendChild(row);
        
        // Add fade-in animation
        row.style.animation = 'fadeIn 0.5s ease-in-out';
    });
}

// Update performance metrics with animations
function updatePerformanceMetrics(data) {
    // Portfolio value with animation
    animateValue('portfolioValue', data.portfolio_value, '$');
    animateValue('cashBalance', data.cash, '$');
    
    // Daily return with color animation
    const dailyReturnElement = document.getElementById('dailyReturn');
    const dailyReturnValue = (data.daily_return * 100).toFixed(2);
    dailyReturnElement.textContent = `${dailyReturnValue}%`;
    dailyReturnElement.className = dailyReturnValue > 0 ? 'positive' : dailyReturnValue < 0 ? 'negative' : '';
    
    // Win rate with animation
    animateValue('winRate', data.win_rate, '', '%');
    
    // Portfolio change
    const portfolioChange = document.getElementById('portfolioChange');
    const changeValue = ((data.portfolio_value / 1000000 - 1) * 100).toFixed(2);
    portfolioChange.textContent = `${changeValue > 0 ? '+' : ''}${changeValue}%`;
    portfolioChange.className = changeValue > 0 ? 'positive' : changeValue < 0 ? 'negative' : '';
}

// Animate value changes with smooth transitions
function animateValue(elementId, newValue, prefix = '', suffix = '') {
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
        })}${suffix}`;
        
        if (currentStep >= steps) {
            clearInterval(interval);
            element.setAttribute('data-value', newValue);
        }
    }, duration / steps);
}

// Update portfolio chart with animations
function updatePortfolioChart(data) {
    // Generate timestamps for the last 30 days
    const timestamps = Array.from({length: 30}, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - (29 - i));
        return date.toLocaleDateString();
    });

    // Calculate portfolio values based on daily returns
    const portfolioValues = data.daily_returns.map((return_value, index) => {
        return data.portfolio_value * (1 + return_value);
    });

    // Update chart with smooth animation
    performanceChart.data.labels = timestamps;
    performanceChart.data.datasets[0].data = portfolioValues;
    performanceChart.update('active');
}

// Update trades table with latest trades
function updateTradesTable(trades) {
    const table = document.getElementById('tradesTable');
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    trades.reverse().forEach((trade, index) => {
        const row = document.createElement('tr');
        const typeClass = trade.type === 'BUY' ? 'buy' : 'sell';
        
        row.innerHTML = `
            <td>${new Date(trade.time).toLocaleString()}</td>
            <td class="font-weight-bold">${trade.ticker}</td>
            <td class="${typeClass}">${trade.type}</td>
            <td>$${parseFloat(trade.price).toFixed(2)}</td>
            <td>${trade.quantity}</td>
            <td>$${(trade.price * trade.quantity).toFixed(2)}</td>
            <td>${trade.status}</td>
        `;
        tbody.appendChild(row);
        
        // Add staggered fade-in animation
        row.style.animation = `fadeIn 0.5s ease-in-out ${index * 0.1}s`;
    });
}

// Update positions table with real-time data
function updatePositionsTable(positions) {
    const table = document.getElementById('positionsTable');
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    Object.entries(positions).forEach(([ticker, position]) => {
        const row = document.createElement('tr');
        const pnlClass = position.unrealized_pnl >= 0 ? 'positive' : 'negative';
        const pnlValue = position.unrealized_pnl || 0;
        
        row.innerHTML = `
            <td class="font-weight-bold">${ticker}</td>
            <td>${position.quantity}</td>
            <td>$${parseFloat(position.price).toFixed(2)}</td>
            <td>$${(position.quantity * position.price).toFixed(2)}</td>
            <td class="${pnlClass}">$${pnlValue.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
        
        // Add fade-in animation
        row.style.animation = 'fadeIn 0.5s ease-in-out';
    });
}

// Update chart time range
function updateChartTimeRange(range) {
    // Implementation will depend on your backend API
    console.log('Updating chart range:', range);
}

// Update last update time with animation
function updateLastUpdateTime() {
    lastUpdateTime = new Date();
    const element = document.getElementById('lastUpdate');
    element.textContent = `Last updated: ${lastUpdateTime.toLocaleString()}`;
    element.style.animation = 'fadeIn 0.5s ease-in-out';
}

// Show info message with animation
function showInfoMessage(message) {
    const alertDiv = document.getElementById('alertMessage');
    alertDiv.textContent = message;
    alertDiv.classList.remove('d-none', 'alert-danger');
    alertDiv.classList.add('alert-info');
    alertDiv.style.animation = 'slideIn 0.3s ease-in-out';
}

// Show error message with animation
function showErrorMessage(message) {
    const alertDiv = document.getElementById('alertMessage');
    alertDiv.textContent = message;
    alertDiv.classList.remove('d-none', 'alert-info');
    alertDiv.classList.add('alert-danger');
    alertDiv.style.animation = 'slideIn 0.3s ease-in-out';
    setTimeout(() => {
        alertDiv.style.animation = 'slideOut 0.3s ease-in-out';
        setTimeout(() => alertDiv.classList.add('d-none'), 300);
    }, 5000);
}

// Fetch with retry logic and timeout
async function fetchWithRetry(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(url, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) return response;
        } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
    }
    throw new Error(`Failed to fetch ${url} after ${retries} retries`);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateY(-100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateY(0); opacity: 1; }
        to { transform: translateY(-100%); opacity: 0; }
    }
    
    .error .status-dot {
        background-color: var(--danger-color);
    }
`;
document.head.appendChild(style);

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeDashboard); 