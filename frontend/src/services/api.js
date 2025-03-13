const API_URL = 'https://gerald-quant-momentum-eaae89eef9a3.herokuapp.com';

export const fetchMomentumSignals = async () => {
    try {
        const response = await fetch(`${API_URL}/api/momentum-signals`);
        if (!response.ok) throw new Error('Failed to fetch momentum signals');
        return await response.json();
    } catch (error) {
        console.error('Error fetching momentum signals:', error);
        throw error;
    }
};

export const fetchPerformanceData = async () => {
    try {
        const response = await fetch(`${API_URL}/api/performance`);
        if (!response.ok) throw new Error('Failed to fetch performance data');
        return await response.json();
    } catch (error) {
        console.error('Error fetching performance data:', error);
        throw error;
    }
};

export const getChartUrl = (filename) => `${API_URL}/api/charts/${filename}`; 