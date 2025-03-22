"""
Module for generating reports and visualizations of momentum trading results.
"""

import pandas as pd
import logging
import os

def generate_report(data: pd.DataFrame, output_path: str) -> None:
    """
    Generate a detailed Excel report of momentum metrics.
    
    Args:
        data (pd.DataFrame): DataFrame containing momentum and risk metrics
        output_path (str): Path to save the Excel report
    """
    try:
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create Excel writer object
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Overview sheet with key metrics
            overview = data[['momentum_score', 'enhanced_score', 'risk_sharpe_ratio', 
                           'risk_sortino_ratio', 'risk_max_drawdown']].copy()
            overview.to_excel(writer, sheet_name='Overview', index=True)
            
            # Returns sheet
            returns = data[['1m_return', '3m_return', '6m_return', '12m_return']].copy()
            returns.to_excel(writer, sheet_name='Returns', index=True)
            
            # Risk metrics sheet
            risk = data[[col for col in data.columns if col.startswith('risk_')]].copy()
            risk.to_excel(writer, sheet_name='Risk Metrics', index=True)
            
            # Technical indicators sheet
            tech = data[['rsi', 'macd', 'macd_signal', 'macd_hist', 
                        'roc_5', 'roc_10', 'roc_20']].copy()
            tech.to_excel(writer, sheet_name='Technical Indicators', index=True)
            
            # Full data sheet
            data.to_excel(writer, sheet_name='Full Data', index=True)
        
        logging.info(f"Report generated successfully at {output_path}")
        
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise 