/**
 * Format a number as currency
 * @param {number} value - The value to format
 * @param {string} [currency='USD'] - The currency code
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

/**
 * Format a number as a percentage
 * @param {number} value - The value to format (e.g., 0.15 for 15%)
 * @param {number} [decimals=2] - Number of decimal places
 * @returns {string} Formatted percentage string
 */
export const formatPercent = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    signDisplay: 'exceptZero'
  }).format(value);
};

/**
 * Format a large number with abbreviations (K, M, B)
 * @param {number} value - The value to format
 * @returns {string} Formatted number string
 */
export const formatLargeNumber = (value: number): string => {
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
export const formatDate = (date: Date | string): string => {
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(date));
};

/**
 * Format a number with fixed decimal places
 * @param {number} value - The value to format
 * @param {number} [decimals=2] - Number of decimal places
 * @returns {string} Formatted number string
 */
export const formatNumber = (value: number, decimals: number = 0): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatVolume = (value: number | null | undefined): string => {
  if (typeof value !== 'number') return 'N/A';
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(2)}B`;
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(2)}M`;
  }
  if (value >= 1e3) {
    return `${(value / 1e3).toFixed(2)}K`;
  }
  return value.toLocaleString();
};

export const formatCompactNumber = (value: number): string => {
  const formatter = new Intl.NumberFormat('en-US', {
    notation: 'compact',
    compactDisplay: 'short',
  });
  return formatter.format(value);
}; 