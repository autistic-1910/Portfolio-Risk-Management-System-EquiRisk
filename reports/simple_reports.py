import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

class SimpleReportGenerator:
    def __init__(self):
        self.color_scheme = {
            'primary': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'danger': '#d13438'
        }
    
    def create_summary_report(self, portfolio_data):
        """Create a simple text-based summary report"""
        report = f"""
        Portfolio Risk Analysis Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Key Metrics:
        - Portfolio Value: ${portfolio_data.get('value', 0):,.2f}
        - VaR (95%): {portfolio_data.get('var_95', 0):.2f}%
        - Sharpe Ratio: {portfolio_data.get('sharpe_ratio', 0):.2f}
        """
        return report