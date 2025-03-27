import { api } from './api';

interface OrderBook {
  bids: [number, number][];
  asks: [number, number][];
  timestamp: number;
}

export interface ArbitrageOpportunity {
  buyExchange: string;
  sellExchange: string;
  symbol: string;
  buyPrice: number;
  sellPrice: number;
  maxSize: number;
  profit: number;
  timestamp: number;
  latency: number;
  confidence: number;
}

class HFTTradingService {
  private orderBooks: { [key: string]: OrderBook } = {};
  private opportunities: ArbitrageOpportunity[] = [];
  private isTrading: boolean = false;
  private minProfitThreshold: number = parseFloat(process.env.REACT_APP_MIN_PROFIT_THRESHOLD || '0.0005');
  private maxLatency: number = parseInt(process.env.REACT_APP_MAX_LATENCY || '50');
  private tradeSizeBTC: number = parseFloat(process.env.REACT_APP_TRADE_SIZE_BTC || '0.01');
  private tradeSizeETH: number = parseFloat(process.env.REACT_APP_TRADE_SIZE_ETH || '0.1');

  constructor() {
    this.initializeWebSockets();
  }

  private initializeWebSockets(): void {
    api.initWebSockets((data: any) => {
      if (data.type === 'orderbook') {
        this.handleOrderBookUpdate(data);
      } else if (data.type === 'arbitrage_opportunity') {
        this.handleArbitrageOpportunity(data);
      }
    });
  }

  private handleOrderBookUpdate(data: any): void {
    const { exchange, symbol, bids, asks, timestamp } = data;
    const key = `${exchange}-${symbol}`;
    
    this.orderBooks[key] = {
      bids: bids.map(([price, size]: [string, string]) => [parseFloat(price), parseFloat(size)]),
      asks: asks.map(([price, size]: [string, string]) => [parseFloat(price), parseFloat(size)]),
      timestamp: parseFloat(timestamp)
    };

    this.detectArbitrageOpportunities();
  }

  private handleArbitrageOpportunity(data: any): void {
    const opportunity: ArbitrageOpportunity = {
      buyExchange: data.buyExchange,
      sellExchange: data.sellExchange,
      symbol: data.symbol,
      buyPrice: parseFloat(data.buyPrice),
      sellPrice: parseFloat(data.sellPrice),
      maxSize: parseFloat(data.maxSize),
      profit: parseFloat(data.profit),
      timestamp: data.timestamp,
      latency: data.latency,
      confidence: data.confidence
    };

    this.opportunities.push(opportunity);
    this.opportunities = this.opportunities.slice(-100); // Keep last 100 opportunities

    if (this.isTrading && this.shouldExecuteTrade(opportunity)) {
      this.executeTrade(opportunity);
    }
  }

  private detectArbitrageOpportunities(): void {
    const symbols = ['BTC-USD', 'ETH-USD'];
    
    for (const symbol of symbols) {
      const coinbaseKey = `coinbase-${symbol}`;
      const binanceKey = `binance-${symbol}`;
      
      if (this.orderBooks[coinbaseKey] && this.orderBooks[binanceKey]) {
        const coinbaseBook = this.orderBooks[coinbaseKey];
        const binanceBook = this.orderBooks[binanceKey];
        
        // Check Coinbase -> Binance
        const cbOpportunity = this.calculateArbitrage(
          'coinbase',
          'binance',
          symbol,
          coinbaseBook.asks[0][0],
          binanceBook.bids[0][0]
        );
        
        if (cbOpportunity) {
          this.handleArbitrageOpportunity(cbOpportunity);
        }
        
        // Check Binance -> Coinbase
        const bcOpportunity = this.calculateArbitrage(
          'binance',
          'coinbase',
          symbol,
          binanceBook.asks[0][0],
          coinbaseBook.bids[0][0]
        );
        
        if (bcOpportunity) {
          this.handleArbitrageOpportunity(bcOpportunity);
        }
      }
    }
  }

  private calculateArbitrage(
    buyExchange: string,
    sellExchange: string,
    symbol: string,
    buyPrice: number,
    sellPrice: number
  ): ArbitrageOpportunity | null {
    const profit = sellPrice - buyPrice;
    const profitPercentage = profit / buyPrice;
    
    if (profitPercentage > this.minProfitThreshold) {
      const maxSize = this.getMaxTradeSize(symbol);
      const totalProfit = profit * maxSize;
      
      return {
        buyExchange,
        sellExchange,
        symbol,
        buyPrice,
        sellPrice,
        maxSize,
        profit: totalProfit,
        timestamp: Date.now(),
        latency: this.getLatency(buyExchange, sellExchange),
        confidence: this.calculateConfidence(profitPercentage)
      };
    }
    
    return null;
  }

  private getMaxTradeSize(symbol: string): number {
    return symbol.includes('BTC') ? this.tradeSizeBTC : this.tradeSizeETH;
  }

  private getLatency(buyExchange: string, sellExchange: string): number {
    // This would be replaced with actual latency measurements
    return Math.random() * 100;
  }

  private calculateConfidence(profitPercentage: number): number {
    // This would be replaced with ML model prediction
    return Math.min(1, profitPercentage / this.minProfitThreshold);
  }

  private shouldExecuteTrade(opportunity: ArbitrageOpportunity): boolean {
    return (
      opportunity.latency <= this.maxLatency &&
      opportunity.confidence >= 0.8 &&
      opportunity.profit > 0
    );
  }

  private async executeTrade(opportunity: ArbitrageOpportunity): Promise<void> {
    try {
      // Execute buy order
      await api.executeOrder({
        exchange: opportunity.buyExchange,
        symbol: opportunity.symbol,
        side: 'BUY',
        price: opportunity.buyPrice,
        size: opportunity.maxSize
      });

      // Execute sell order
      await api.executeOrder({
        exchange: opportunity.sellExchange,
        symbol: opportunity.symbol,
        side: 'SELL',
        price: opportunity.sellPrice,
        size: opportunity.maxSize
      });

      console.log(`Executed arbitrage trade: ${JSON.stringify(opportunity)}`);
    } catch (error) {
      console.error('Error executing trade:', error);
    }
  }

  public startTrading(): void {
    this.isTrading = true;
    console.log('HFT trading started');
  }

  public stopTrading(): void {
    this.isTrading = false;
    console.log('HFT trading stopped');
  }

  public getOpportunities(): ArbitrageOpportunity[] {
    return this.opportunities;
  }

  public getOrderBooks(): { [key: string]: OrderBook } {
    return this.orderBooks;
  }
}

export const hftTrading = new HFTTradingService();
export default hftTrading; 