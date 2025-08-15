import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from jinja2 import Template
from typing import Dict  # Add this line

class ModernReportGenerator:
    def __init__(self):
        self.color_scheme = {
            'primary': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'danger': '#d13438',
            'background': '#f8f9fa'
        }
    
    def create_executive_summary(self, portfolio_data: Dict) -> str:
        template = Template("""
        #  Portfolio Risk Analysis - Executive Summary
        
        ## Key Metrics
        | Metric | Value | Status |
        |--------|-------|--------|
        | Portfolio Value | ${{ portfolio_value:,.2f }} | {{ value_status }} |
        | VaR (95%) | {{ var_95 }}% | {{ var_status }} |
        | Expected Shortfall | {{ es_95 }}% | {{ es_status }} |
        | Sharpe Ratio | {{ sharpe_ratio:.2f }} | {{ sharpe_status }} |
        
        ## Risk Assessment
        {{ risk_assessment }}
        
        ## Recommendations
        {{ recommendations }}
        """)
        
        return template.render(**portfolio_data)
    
    def create_interactive_dashboard(self, data: Dict) -> go.Figure:
        # Create comprehensive dashboard with multiple subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Composition', 'Risk Metrics Over Time',
                'Return Distribution', 'Stress Test Results',
                'Correlation Heatmap', 'Performance Attribution'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "waterfall"}]
            ]
        )
        
        # Add portfolio composition pie chart
        fig.add_trace(
            go.Pie(
                labels=data['symbols'],
                values=data['weights'],
                name="Portfolio Composition",
                marker_colors=px.colors.qualitative.Set3
            ),
            row=1, col=1
        )
        
        # Add risk metrics timeline
        fig.add_trace(
            go.Scatter(
                x=data['dates'],
                y=data['var_timeline'],
                mode='lines+markers',
                name='VaR (95%)',
                line=dict(color=self.color_scheme['danger'])
            ),
            row=1, col=2
        )
        
        # Update layout for professional appearance
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Portfolio Risk Analysis Dashboard",
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def generate_pdf_report(self, data: Dict, filename: str):
        """Generate professional PDF report using reportlab"""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor='#0078d4'
        )
        
        story.append(Paragraph("Portfolio Risk Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Executive summary table
        summary_data = [
            ['Metric', 'Value', 'Benchmark', 'Status'],
            ['Portfolio VaR (95%)', f"{data['var_95']:.2f}%", '< 3.0%', 'PASS' if data['var_95'] < 3.0 else 'WARNING'],
            ['Sharpe Ratio', f"{data['sharpe_ratio']:.2f}", '> 1.0', 'PASS' if data['sharpe_ratio'] > 1.0 else 'WARNING'],
            ['Max Drawdown', f"{data['max_drawdown']:.2f}%", '< 10%', 'PASS' if abs(data['max_drawdown']) < 10 else 'WARNING']
        ]
        
        summary_table = Table(summary_data)
        story.append(summary_table)
        
        doc.build(story)