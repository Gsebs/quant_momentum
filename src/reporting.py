"""
Reporting module for generating detailed momentum strategy reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import logging
from . import config

logger = logging.getLogger(__name__)

class MomentumReport:
    def __init__(self, output_dir: str = "data/reports"):
        """Initialize the reporting module."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def _format_momentum_sheet(self, writer: pd.ExcelWriter, momentum_df: pd.DataFrame, sheet_name: str = "Momentum Signals"):
        """Format the momentum signals sheet with conditional formatting and charts."""
        # Write data
        momentum_df.to_excel(writer, sheet_name=sheet_name)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D9E1F2',
            'border': 1
        })
        
        # Create conditional formatting
        worksheet.conditional_format('D2:G100', {
            'type': '3_color_scale',
            'min_color': "#FF6B6B",
            'mid_color': "#FFFFFF",
            'max_color': "#4ECB71"
        })
        
        # Adjust column widths
        worksheet.set_column('A:A', 12)  # Ticker
        worksheet.set_column('B:B', 15)  # Price
        worksheet.set_column('C:K', 12)  # Metrics
        
        # Add header row
        for col_num, value in enumerate(momentum_df.columns.values):
            worksheet.write(0, col_num + 1, value, header_format)
    
    def _create_performance_chart(self, momentum_df: pd.DataFrame) -> None:
        """Create performance visualization charts."""
        plt.figure(figsize=(12, 6))
        
        # Plot returns across different timeframes
        periods = ['1m_momentum', '3m_momentum', '6m_momentum', '12m_momentum']
        data = momentum_df[periods].head(10)
        
        ax = data.plot(kind='bar', width=0.8)
        plt.title('Returns Across Different Timeframes (Top 10 Stocks)')
        plt.xlabel('Stocks')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(os.path.join(self.output_dir, 'momentum_performance.png'))
        plt.close()
    
    def _create_correlation_heatmap(self, momentum_df: pd.DataFrame) -> None:
        """Create correlation heatmap of momentum metrics."""
        plt.figure(figsize=(10, 8))
        
        # Select momentum and ML columns
        cols = [col for col in momentum_df.columns if 'momentum' in col or 'score' in col]
        corr = momentum_df[cols].corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation of Momentum Metrics')
        plt.tight_layout()
        
        # Save chart
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()
    
    def generate_report(self, 
                       momentum_df: pd.DataFrame,
                       recommendations: pd.DataFrame,
                       output_file: str = "momentum_report.xlsx") -> None:
        """
        Generate a detailed Excel report with multiple sheets and visualizations.
        
        Args:
            momentum_df: DataFrame with momentum metrics
            recommendations: DataFrame with trade recommendations
            output_file: Name of the output Excel file
        """
        try:
            output_path = os.path.join(self.output_dir, output_file)
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Summary sheet
                summary = pd.DataFrame({
                    'Metric': [
                        'Report Date',
                        'Number of Stocks Analyzed',
                        'Average Momentum Score',
                        'Average ML Score',
                        'Top Performing Stock',
                        'Best 1-Month Return',
                        'Best 12-Month Return'
                    ],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d'),
                        len(momentum_df),
                        f"{momentum_df['composite_score'].mean():.2%}",
                        f"{momentum_df['ml_score'].mean():.2%}",
                        momentum_df.index[0],
                        f"{momentum_df['1m_momentum'].max():.2%}",
                        f"{momentum_df['12m_momentum'].max():.2%}"
                    ]
                })
                
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                workbook = writer.book
                worksheet = writer.sheets['Summary']
                header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:B', 20)
                
                # Recommendations sheet
                recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Detailed momentum metrics sheet
                self._format_momentum_sheet(writer, momentum_df)
                
                # Create and save visualizations
                self._create_performance_chart(momentum_df)
                self._create_correlation_heatmap(momentum_df)
                
                # Add charts to Excel
                chart_sheet = workbook.add_worksheet('Charts')
                chart_sheet.insert_image('A1', os.path.join(self.output_dir, 'momentum_performance.png'))
                chart_sheet.insert_image('A30', os.path.join(self.output_dir, 'correlation_heatmap.png'))
            
            logger.info(f"Report generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}") 