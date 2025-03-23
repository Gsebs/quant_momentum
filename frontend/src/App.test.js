import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

// Mock fetch
global.fetch = jest.fn();

// Mock API calls
jest.mock('./services/api', () => ({
  get: jest.fn().mockResolvedValue({ data: {} }),
}));

describe('App', () => {
  beforeEach(() => {
    global.fetch.mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('renders app title', () => {
    render(<App />);
    const titleElement = screen.getByText(/Quant Momentum Strategy/i);
    expect(titleElement).toBeInTheDocument();
  });
});
