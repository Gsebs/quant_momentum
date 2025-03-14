/**
 * Format a number as currency
 * @param {number} value - The value to format
 * @param {string} [currency='USD'] - The currency code
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value, currency = 'USD') => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * Format a number as a percentage
 * @param {number} value - The value to format (e.g., 0.15 for 15%)
 * @param {number} [decimals=2] - Number of decimal places
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value, decimals = 2) => {
  return `${(value * 100).toFixed(decimals)}%`;
};

/**
 * Format a large number with abbreviations (K, M, B)
 * @param {number} value - The value to format
 * @returns {string} Formatted number string
 */
export const formatLargeNumber = (value) => {
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(1)}B`;
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(1)}M`;
  }
  if (value >= 1e3) {
    return `${(value / 1e3).toFixed(1)}K`;
  }
  return value.toString();
};

/**
 * Format a date to a readable string
 * @param {Date|string} date - The date to format
 * @param {boolean} [includeTime=true] - Whether to include the time
 * @returns {string} Formatted date string
 */
export const formatDate = (date, includeTime = true) => {
  const options = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...(includeTime && {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }),
  };
  
  return new Date(date).toLocaleString('en-US', options);
};

/**
 * Format a number with fixed decimal places
 * @param {number} value - The value to format
 * @param {number} [decimals=2] - Number of decimal places
 * @returns {string} Formatted number string
 */
export const formatNumber = (value, decimals = 2) => {
  return Number(value).toFixed(decimals);
}; 