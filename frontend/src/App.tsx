import React, { useState, useEffect, useRef } from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  Grid,
  theme,
  Heading,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  useToast,
  HStack,
  Progress,
  CircularProgress,
  CircularProgressLabel,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Card,
  CardHeader,
  CardBody
} from '@chakra-ui/react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, AreaChart, Area } from 'recharts';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

interface Trade {
  timestamp: string;
  symbol: string;
  side: string;
  price: number;
  quantity: number;
  profit: number;
  latency_ms: number;
}

interface Alert {
  timestamp: string;
  level: string;
  message: string;
}

interface Metrics {
  total_profit: number;
  win_rate: number;
  total_trades: number;
  avg_latency_ms: number;
  last_update: string;
  sharpe_ratio?: number;
  max_drawdown?: number;
  volatility?: number;
  success_rate?: number;
  avg_profit_per_trade?: number;
}

interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  bid: number;
  ask: number;
  spread: number;
  timestamp: string;
}

function App() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    total_profit: 0,
    win_rate: 0,
    total_trades: 0,
    avg_latency_ms: 0,
    last_update: '',
    sharpe_ratio: 0,
    max_drawdown: 0,
    volatility: 0,
    success_rate: 0,
    avg_profit_per_trade: 0
  });
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const toast = useToast();

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        setIsConnected(true);
        toast({
          title: 'Connected to server',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      };

      ws.onclose = () => {
        setIsConnected(false);
        toast({
          title: 'Disconnected from server',
          status: 'warning',
          duration: 3000,
          isClosable: true,
        });
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case 'trade':
            setTrades(prev => [...prev, data.data].slice(-1000));
            break;
          case 'alert':
            setAlerts(prev => [...prev, data.data].slice(-100));
            break;
          case 'metrics':
            setMetrics(data.data);
            break;
          case 'market_data':
            setMarketData(prev => [...prev, data.data].slice(-1000));
            break;
        }
      };

      wsRef.current = ws;
    };

    connectWebSocket();
    return () => wsRef.current?.close();
  }, [toast]);

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [tradesRes, alertsRes, metricsRes, marketDataRes] = await Promise.all([
          fetch(`${API_URL}/trades`),
          fetch(`${API_URL}/alerts`),
          fetch(`${API_URL}/metrics`),
          fetch(`${API_URL}/market-data`)
        ]);
        
        const [tradesData, alertsData, metricsData, marketDataData] = await Promise.all([
          tradesRes.json(),
          alertsRes.json(),
          metricsRes.json(),
          marketDataRes.json()
        ]);
        
        setTrades(tradesData);
        setAlerts(alertsData);
        setMetrics(metricsData);
        setMarketData(Object.values(marketDataData));
        
      } catch (error) {
        console.error('Error fetching data:', error);
        toast({
          title: 'Error fetching data',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };
    
    fetchData();
  }, [toast]);

  return (
    <ChakraProvider theme={theme}>
      <Box textAlign="center" fontSize="xl" p={5}>
        <VStack spacing={8}>
          {/* Header */}
          <HStack w="100%" justify="space-between">
            <Heading as="h1" size="2xl">HFT System Monitor</Heading>
            <Badge
              colorScheme={isConnected ? 'green' : 'red'}
              p={2}
              borderRadius="md"
            >
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
          </HStack>

          {/* Main Content */}
          <Tabs variant="enclosed" w="100%">
            <TabList>
              <Tab>Overview</Tab>
              <Tab>Trading</Tab>
              <Tab>Performance</Tab>
              <Tab>Alerts</Tab>
            </TabList>

            <TabPanels>
              {/* Overview Panel */}
              <TabPanel>
                <VStack spacing={6}>
                  {/* Key Metrics */}
                  <Grid templateColumns="repeat(4, 1fr)" gap={6} w="100%">
                    <Card>
                      <CardHeader>
                        <StatLabel>Total Profit</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <StatNumber>
                          ${metrics.total_profit.toFixed(2)}
                          <StatArrow type={metrics.total_profit >= 0 ? 'increase' : 'decrease'} />
                        </StatNumber>
                        <StatHelpText>
                          Win Rate: {(metrics.win_rate * 100).toFixed(1)}%
                        </StatHelpText>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader>
                        <StatLabel>Performance</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <CircularProgress value={metrics.success_rate * 100 || 0} color="green.400">
                          <CircularProgressLabel>
                            {(metrics.success_rate * 100 || 0).toFixed(1)}%
                          </CircularProgressLabel>
                        </CircularProgress>
                        <StatHelpText>Success Rate</StatHelpText>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader>
                        <StatLabel>Latency</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <StatNumber>{metrics.avg_latency_ms.toFixed(2)}ms</StatNumber>
                        <Progress
                          value={Math.min((metrics.avg_latency_ms / 50) * 100, 100)}
                          colorScheme={metrics.avg_latency_ms < 50 ? 'green' : 'red'}
                        />
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader>
                        <StatLabel>Risk Metrics</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <VStack>
                          <Text>Sharpe: {metrics.sharpe_ratio?.toFixed(2) || 'N/A'}</Text>
                          <Text>DrawDown: {metrics.max_drawdown?.toFixed(2)}%</Text>
                        </VStack>
                      </CardBody>
                    </Card>
                  </Grid>

                  {/* Charts */}
                  <Grid templateColumns="repeat(2, 1fr)" gap={6} w="100%">
                    <Box bg="gray.50" borderRadius="lg" p={4}>
                      <Heading size="md" mb={4}>Profit History</Heading>
                      <AreaChart width={600} height={300} data={trades}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <Tooltip />
                        <Area type="monotone" dataKey="profit" stroke="#8884d8" fill="#8884d8" />
                      </AreaChart>
                    </Box>

                    <Box bg="gray.50" borderRadius="lg" p={4}>
                      <Heading size="md" mb={4}>Latency Distribution</Heading>
                      <LineChart width={600} height={300} data={trades}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="latency_ms" stroke="#82ca9d" />
                      </LineChart>
                    </Box>
                  </Grid>
                </VStack>
              </TabPanel>

              {/* Trading Panel */}
              <TabPanel>
                <VStack spacing={6}>
                  {/* Market Data */}
                  <Box w="100%">
                    <Heading size="md" mb={4}>Market Data</Heading>
                    <Table variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Symbol</Th>
                          <Th>Price</Th>
                          <Th>Spread</Th>
                          <Th>Volume</Th>
                          <Th>Time</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {marketData.slice(-5).map((data, i) => (
                          <Tr key={i}>
                            <Td>{data.symbol}</Td>
                            <Td>${data.price.toFixed(2)}</Td>
                            <Td>{data.spread.toFixed(4)}</Td>
                            <Td>{data.volume.toFixed(4)}</Td>
                            <Td>{new Date(data.timestamp).toLocaleTimeString()}</Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  </Box>

                  {/* Recent Trades */}
                  <Box w="100%">
                    <Heading size="md" mb={4}>Recent Trades</Heading>
                    <Table variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Time</Th>
                          <Th>Symbol</Th>
                          <Th>Side</Th>
                          <Th>Price</Th>
                          <Th>Quantity</Th>
                          <Th>Profit</Th>
                          <Th>Latency</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {trades.slice(-10).map((trade, i) => (
                          <Tr key={i}>
                            <Td>{new Date(trade.timestamp).toLocaleTimeString()}</Td>
                            <Td>{trade.symbol}</Td>
                            <Td>
                              <Badge colorScheme={trade.side === 'buy' ? 'green' : 'red'}>
                                {trade.side}
                              </Badge>
                            </Td>
                            <Td>${trade.price.toFixed(2)}</Td>
                            <Td>{trade.quantity.toFixed(4)}</Td>
                            <Td>
                              <Text color={trade.profit >= 0 ? 'green.500' : 'red.500'}>
                                ${trade.profit.toFixed(2)}
                              </Text>
                            </Td>
                            <Td>{trade.latency_ms.toFixed(2)}ms</Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  </Box>
                </VStack>
              </TabPanel>

              {/* Performance Panel */}
              <TabPanel>
                <VStack spacing={6}>
                  <Grid templateColumns="repeat(3, 1fr)" gap={6} w="100%">
                    <Card>
                      <CardHeader>
                        <StatLabel>Average Profit per Trade</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <StatNumber>
                          ${metrics.avg_profit_per_trade?.toFixed(2) || '0.00'}
                        </StatNumber>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader>
                        <StatLabel>Volatility</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <StatNumber>
                          {(metrics.volatility * 100)?.toFixed(2) || '0.00'}%
                        </StatNumber>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader>
                        <StatLabel>Total Trades</StatLabel>
                      </CardHeader>
                      <CardBody>
                        <StatNumber>{metrics.total_trades}</StatNumber>
                      </CardBody>
                    </Card>
                  </Grid>

                  {/* Performance Charts */}
                  <Box w="100%" h="400px" bg="gray.50" borderRadius="lg" p={4}>
                    <Heading size="md" mb={4}>Cumulative Performance</Heading>
                    <AreaChart width={1200} height={300} data={trades}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="profit" stroke="#8884d8" fill="#8884d8" />
                    </AreaChart>
                  </Box>
                </VStack>
              </TabPanel>

              {/* Alerts Panel */}
              <TabPanel>
                <Box w="100%">
                  <Heading size="md" mb={4}>System Alerts</Heading>
                  <Table variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Time</Th>
                        <Th>Level</Th>
                        <Th>Message</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {alerts.slice(-20).map((alert, i) => (
                        <Tr key={i}>
                          <Td>{new Date(alert.timestamp).toLocaleTimeString()}</Td>
                          <Td>
                            <Badge
                              colorScheme={
                                alert.level === 'error' ? 'red' :
                                alert.level === 'warning' ? 'yellow' : 'green'
                              }
                            >
                              {alert.level}
                            </Badge>
                          </Td>
                          <Td>{alert.message}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </VStack>
      </Box>
    </ChakraProvider>
  );
}

export default App; 