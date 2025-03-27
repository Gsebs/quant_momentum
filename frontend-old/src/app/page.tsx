'use client';

import { useEffect, useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Progress } from '../components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { WebSocketManager } from '../lib/api';
import { HFTTradingService } from '../lib/hftTrading';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TradeOpportunity {
  exchange1: string;
  exchange2: string;
  symbol: string;
  profitPercentage: number;
  timestamp: string;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  amount: number;
  timestamp: string;
  exchange: string;
}

interface WebSocketMessage {
  type: string;
  opportunity?: TradeOpportunity;
  trade?: Trade;
  latency?: number;
  profit?: number;
}

export default function Dashboard() {
  const [opportunities, setOpportunities] = useState<TradeOpportunity[]>([]);
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [isTrading, setIsTrading] = useState(false);
  const [profitHistory, setProfitHistory] = useState<number[]>([]);
  const [totalProfit, setTotalProfit] = useState(0);
  const [latency, setLatency] = useState(0);
  const hftServiceRef = useRef<HFTTradingService | null>(null);
  const wsManagerRef = useRef<WebSocketManager | null>(null);

  useEffect(() => {
    const wsManager = new WebSocketManager();
    const hftService = new HFTTradingService();

    wsManagerRef.current = wsManager;
    hftServiceRef.current = hftService;

    wsManager.connect();

    const cleanup = wsManager.onMessage((data: WebSocketMessage) => {
      if (data.type === 'opportunity' && data.opportunity) {
        setOpportunities((prev) => {
          const newOpportunities = [...prev, data.opportunity!]
          return newOpportunities.slice(-10)
        });
      } else if (data.type === 'trade' && data.trade) {
        setRecentTrades((prev) => {
          const newTrades = [...prev, data.trade!]
          return newTrades.slice(-20)
        });
        const profit = data.profit;
        if (typeof profit === 'number') {
          setProfitHistory((prev) => [...prev, profit])
          setTotalProfit((prev) => prev + profit)
        }
      } else if (data.type === 'latency' && typeof data.latency === 'number') {
        setLatency(data.latency)
      }
    });

    return () => {
      wsManager.disconnect();
      cleanup();
    };
  }, []);

  const handleStartTrading = () => {
    const wsManager = wsManagerRef.current;
    const hftService = hftServiceRef.current;
    const ws = wsManager?.getWebSocket();

    if (hftService && ws) {
      hftService.setWebSocketManager(ws);
      hftService.startTrading();
      setIsTrading(true);
    }
  };

  const handleStopTrading = () => {
    const hftService = hftServiceRef.current;
    if (hftService) {
      hftService.stopTrading();
      setIsTrading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Trading Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <span className={isTrading ? 'text-green-500' : 'text-red-500'}>
                {isTrading ? 'Active' : 'Inactive'}
              </span>
              <Button onClick={isTrading ? handleStopTrading : handleStartTrading}>
                {isTrading ? 'Stop Trading' : 'Start Trading'}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Total Profit</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${totalProfit.toFixed(2)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-xl">{latency.toFixed(2)}ms</div>
              <Progress value={Math.min((latency / 100) * 100, 100)} />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8">
        <Tabs defaultValue="opportunities">
          <TabsList>
            <TabsTrigger value="opportunities">Opportunities</TabsTrigger>
            <TabsTrigger value="trades">Recent Trades</TabsTrigger>
          </TabsList>
          <TabsContent value="opportunities">
            <Card>
              <CardHeader>
                <CardTitle>Latest Opportunities</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Exchange Pair</TableHead>
                      <TableHead>Profit %</TableHead>
                      <TableHead>Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {opportunities.map((opp, i) => (
                      <TableRow key={i}>
                        <TableCell>{opp.symbol}</TableCell>
                        <TableCell>{`${opp.exchange1} → ${opp.exchange2}`}</TableCell>
                        <TableCell className="text-green-500">
                          {opp.profitPercentage.toFixed(2)}%
                        </TableCell>
                        <TableCell>{new Date(opp.timestamp).toLocaleTimeString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="trades">
            <Card>
              <CardHeader>
                <CardTitle>Recent Trades</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead>Price</TableHead>
                      <TableHead>Amount</TableHead>
                      <TableHead>Exchange</TableHead>
                      <TableHead>Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentTrades.map((trade) => (
                      <TableRow key={trade.id}>
                        <TableCell>{trade.symbol}</TableCell>
                        <TableCell className={trade.side === 'buy' ? 'text-green-500' : 'text-red-500'}>
                          {trade.side.toUpperCase()}
                        </TableCell>
                        <TableCell>${trade.price.toFixed(2)}</TableCell>
                        <TableCell>{trade.amount.toFixed(8)}</TableCell>
                        <TableCell>{trade.exchange}</TableCell>
                        <TableCell>{new Date(trade.timestamp).toLocaleTimeString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
} 