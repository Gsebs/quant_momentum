from setuptools import setup, find_packages

setup(
    name="quant_momentum",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'yfinance>=0.1.63',
        'matplotlib>=3.4.0',
        'scikit-learn>=0.24.0',
        'scipy>=1.7.0',
        'jupyter>=1.0.0',
        'nbformat>=5.1.3',
        'openpyxl>=3.0.7',
        'requests>=2.26.0',
        'beautifulsoup4>=4.9.3',
        'ta>=0.7.0',
        'pytest>=6.2.5'
    ],
) 