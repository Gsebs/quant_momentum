'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card';
import { testSystem } from '../../lib/test';

interface TestResults {
  wsConnected: boolean;
  hasOrderBooks: boolean;
  hasOpportunities: boolean;
  tradingStatus: boolean;
}

export default function TestPage() {
  const [results, setResults] = useState<TestResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const runTests = async () => {
      try {
        const testResults = await testSystem();
        setResults(testResults);
      } catch (error) {
        console.error('Test failed:', error);
      } finally {
        setLoading(false);
      }
    };

    runTests();
  }, []);

  if (loading) {
    return <div className="container mx-auto p-4">Running system tests...</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">System Test Results</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>WebSocket Connection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-lg ${results?.wsConnected ? 'text-green-500' : 'text-red-500'}`}>
              {results?.wsConnected ? 'Connected' : 'Disconnected'}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Order Books</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-lg ${results?.hasOrderBooks ? 'text-green-500' : 'text-red-500'}`}>
              {results?.hasOrderBooks ? 'Receiving Updates' : 'No Updates'}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Arbitrage Opportunities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-lg ${results?.hasOpportunities ? 'text-green-500' : 'text-red-500'}`}>
              {results?.hasOpportunities ? 'Detected' : 'None Detected'}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Trading Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-lg ${results?.tradingStatus ? 'text-green-500' : 'text-yellow-500'}`}>
              {results?.tradingStatus ? 'Active' : 'Inactive'}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 