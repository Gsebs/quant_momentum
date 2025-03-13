import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';

const PerformanceChart = ({ data }) => {
    if (!data || data.length === 0) {
        return <div>No performance data available</div>;
    }

    return (
        <ResponsiveContainer width="100%" height={400}>
            <LineChart
                data={data}
                margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                />
                <YAxis 
                    tick={{ fontSize: 12 }}
                    label={{ 
                        value: 'Portfolio Value ($)', 
                        angle: -90, 
                        position: 'insideLeft',
                        style: { textAnchor: 'middle' }
                    }}
                />
                <Tooltip />
                <Legend />
                <Line
                    type="monotone"
                    dataKey="portfolio_value"
                    name="Portfolio Value"
                    stroke="#8884d8"
                    dot={false}
                />
                <Line
                    type="monotone"
                    dataKey="benchmark_value"
                    name="Benchmark (S&P 500)"
                    stroke="#82ca9d"
                    dot={false}
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default PerformanceChart; 