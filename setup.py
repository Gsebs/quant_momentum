from setuptools import setup, find_packages

setup(
    name="quant_momentum",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==2.0.1',
        'Werkzeug==2.0.1',
        'Flask-Cors==3.0.10',
        'gunicorn==20.1.0',
        'pandas==1.5.3',
        'yfinance==0.2.18',
        'numpy==1.24.2',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'openpyxl==3.1.2',
        'redis==4.3.4',
        'Flask-Limiter==2.4.0',
        'requests-cache==1.1.1',
        'python-dotenv==1.0.0',
        'scikit-learn==1.2.2',
        'aiohttp==3.11.13'
    ],
) 