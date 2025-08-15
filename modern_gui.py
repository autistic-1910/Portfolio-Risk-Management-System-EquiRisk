import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QTabWidget, QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QProgressBar, QSplitter,
    QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QGridLayout, QMessageBox, QFileDialog, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class MonteCarloWorker(QThread):
    """Worker thread for Monte Carlo simulation"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, returns_data, num_simulations, time_horizon):
        super().__init__()
        self.returns_data = returns_data
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
    
    def run(self):
        try:
            results = self.run_monte_carlo()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
    
    def run_monte_carlo(self):
        """Run Monte Carlo simulation"""
        returns_df = pd.DataFrame(self.returns_data)
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Equal weights for simplicity
        weights = np.array([1/len(returns_df.columns)] * len(returns_df.columns))
        
        # Portfolio statistics
        portfolio_mean = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Monte Carlo simulation
        simulated_returns = []
        
        for i in range(self.num_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, self.time_horizon
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(random_returns * weights, axis=1)
            cumulative_return = np.prod(1 + portfolio_returns) - 1
            simulated_returns.append(cumulative_return)
            
            # Update progress
            progress = int((i + 1) / self.num_simulations * 100)
            self.progress.emit(progress)
        
        # Calculate statistics
        simulated_returns = np.array(simulated_returns)
        
        return {
            'simulated_returns': simulated_returns,
            'mean_return': np.mean(simulated_returns),
            'std_return': np.std(simulated_returns),
            'var_95': np.percentile(simulated_returns, 5),
            'var_99': np.percentile(simulated_returns, 1),
            'portfolio_mean': portfolio_mean,
            'portfolio_std': portfolio_std
        }

class OptimizationWorker(QThread):
    """Worker thread for portfolio optimization"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, returns_data, risk_free_rate=0.02):
        super().__init__()
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
    
    def run(self):
        try:
            results = self.optimize_portfolio()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
    
    def optimize_portfolio(self):
        """Optimize portfolio using mean-variance optimization"""
        returns_df = pd.DataFrame(self.returns_data)
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        num_assets = len(mean_returns)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Efficient frontier
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
        efficient_portfolios = []
        
        for i, target in enumerate(target_returns):
            # Constraint for target return
            cons = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target: np.sum(x * mean_returns) - target}
            ]
            
            # Minimize portfolio variance
            result = minimize(
                lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                num_assets * [1. / num_assets],
                method='SLSQP',
                bounds=bounds,
                constraints=cons
            )
            
            if result.success:
                efficient_portfolios.append({
                    'return': target,
                    'risk': result.fun,
                    'weights': result.x
                })
            
            # Update progress
            progress = int((i + 1) / len(target_returns) * 100)
            self.progress.emit(progress)
        
        # Find optimal portfolios
        # Maximum Sharpe ratio
        def neg_sharpe(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
        
        max_sharpe_result = minimize(
            neg_sharpe,
            num_assets * [1. / num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Minimum variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        min_var_result = minimize(
            portfolio_variance,
            num_assets * [1. / num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'efficient_frontier': efficient_portfolios,
            'max_sharpe': {
                'weights': max_sharpe_result.x,
                'return': np.sum(max_sharpe_result.x * mean_returns),
                'risk': np.sqrt(np.dot(max_sharpe_result.x.T, np.dot(cov_matrix, max_sharpe_result.x))),
                'sharpe': -max_sharpe_result.fun
            },
            'min_variance': {
                'weights': min_var_result.x,
                'return': np.sum(min_var_result.x * mean_returns),
                'risk': np.sqrt(min_var_result.fun)
            },
            'asset_names': list(returns_df.columns)
        }

class DataWorker(QThread):
    """Worker thread for data loading"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, symbols, start_date, end_date):
        super().__init__()
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
    
    def run(self):
        try:
            data = {}
            for i, symbol in enumerate(self.symbols):
                ticker = yf.Ticker(symbol)
                stock_data = ticker.history(start=self.start_date, end=self.end_date)
                
                # Validate data
                if stock_data.empty:
                    self.error.emit(f"No data found for {symbol}")
                    return
                
                # Check if we have reasonable amount of data
                if len(stock_data) < 10:
                    self.error.emit(f"Insufficient data for {symbol}: only {len(stock_data)} days")
                    return
                
                data[symbol] = stock_data
                
                progress = int((i + 1) / len(self.symbols) * 100)
                self.progress.emit(progress)
            
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))

class ModernRiskManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Portfolio Risk Manager")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize data
        self.portfolio_data = {}
        self.risk_metrics = {}
        self.monte_carlo_results = {}
        self.optimization_results = {}
        
        # Initialize database
        self.init_database()
        
        # Set up UI
        self.init_ui()
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('portfolio_data.db')
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                symbol TEXT,
                price REAL,
                volume INTEGER
            )
        ''')
        
        self.conn.commit()
    
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_data_input_tab()
        self.create_analysis_tab()
        self.create_monte_carlo_tab()
        self.create_optimization_tab()
        self.create_visualization_tab()
        self.create_reports_tab()
    
    def create_data_input_tab(self):
        """Create data input tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input group
        input_group = QGroupBox("Portfolio Data Input")
        input_layout = QGridLayout(input_group)
        
        # Symbols input
        input_layout.addWidget(QLabel("Stock Symbols (comma-separated):"), 0, 0)
        self.symbols_input = QLineEdit("AAPL,GOOGL,MSFT,TSLA")
        input_layout.addWidget(self.symbols_input, 0, 1)
        
        # Date inputs
        input_layout.addWidget(QLabel("Start Date (YYYY-MM-DD):"), 1, 0)
        self.start_date_input = QLineEdit("2020-01-01")
        input_layout.addWidget(self.start_date_input, 1, 1)
        
        input_layout.addWidget(QLabel("End Date (YYYY-MM-DD):"), 2, 0)
        self.end_date_input = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        input_layout.addWidget(self.end_date_input, 2, 1)
        
        # Load button
        self.load_button = QPushButton("Load Portfolio Data")
        self.load_button.clicked.connect(self.load_data)
        input_layout.addWidget(self.load_button, 3, 0, 1, 2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        input_layout.addWidget(self.progress_bar, 4, 0, 1, 2)
        
        layout.addWidget(input_group)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.data_table = QTableWidget()
        preview_layout.addWidget(self.data_table)
        
        layout.addWidget(preview_group)
        
        self.tab_widget.addTab(tab, "Data Input")
    
    def create_analysis_tab(self):
        """Create risk analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis controls
        controls_group = QGroupBox("Risk Analysis Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Confidence level
        controls_layout.addWidget(QLabel("Confidence Level:"), 0, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.90, 0.99)
        self.confidence_spin.setValue(0.95)
        self.confidence_spin.setSingleStep(0.01)
        controls_layout.addWidget(self.confidence_spin, 0, 1)
        
        # Time horizon
        controls_layout.addWidget(QLabel("Time Horizon (days):"), 1, 0)
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 252)
        self.horizon_spin.setValue(22)
        controls_layout.addWidget(self.horizon_spin, 1, 1)
        
        # Calculate button
        self.calculate_button = QPushButton("Calculate VaR & Risk Metrics")
        self.calculate_button.clicked.connect(self.calculate_var)
        controls_layout.addWidget(self.calculate_button, 2, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        self.tab_widget.addTab(tab, "Risk Analysis")
    
    def create_monte_carlo_tab(self):
        """Create Monte Carlo simulation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Monte Carlo controls
        controls_group = QGroupBox("Monte Carlo Simulation")
        controls_layout = QGridLayout(controls_group)
        
        # Number of simulations
        controls_layout.addWidget(QLabel("Number of Simulations:"), 0, 0)
        self.mc_simulations_spin = QSpinBox()
        self.mc_simulations_spin.setRange(1000, 100000)
        self.mc_simulations_spin.setValue(10000)
        self.mc_simulations_spin.setSingleStep(1000)
        controls_layout.addWidget(self.mc_simulations_spin, 0, 1)
        
        # Time horizon
        controls_layout.addWidget(QLabel("Time Horizon (days):"), 1, 0)
        self.mc_horizon_spin = QSpinBox()
        self.mc_horizon_spin.setRange(1, 252)
        self.mc_horizon_spin.setValue(22)
        controls_layout.addWidget(self.mc_horizon_spin, 1, 1)
        
        # Run simulation button
        self.mc_button = QPushButton("Run Monte Carlo Simulation")
        self.mc_button.clicked.connect(self.run_monte_carlo)
        controls_layout.addWidget(self.mc_button, 2, 0, 1, 2)
        
        # Progress bar
        self.mc_progress = QProgressBar()
        controls_layout.addWidget(self.mc_progress, 3, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Results
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Plot widget
        self.mc_plot_widget = PlotWidget()
        self.mc_plot_widget.setBackground('w')
        results_splitter.addWidget(self.mc_plot_widget)
        
        # Results text
        self.mc_results_text = QTextEdit()
        self.mc_results_text.setReadOnly(True)
        results_splitter.addWidget(self.mc_results_text)
        
        layout.addWidget(results_splitter)
        
        self.tab_widget.addTab(tab, "Monte Carlo")
    
    def create_optimization_tab(self):
        """Create portfolio optimization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Optimization controls
        controls_group = QGroupBox("Portfolio Optimization")
        controls_layout = QGridLayout(controls_group)
        
        # Risk-free rate
        controls_layout.addWidget(QLabel("Risk-Free Rate:"), 0, 0)
        self.risk_free_spin = QDoubleSpinBox()
        self.risk_free_spin.setRange(0.0, 0.1)
        self.risk_free_spin.setValue(0.02)
        self.risk_free_spin.setSingleStep(0.001)
        controls_layout.addWidget(self.risk_free_spin, 0, 1)
        
        # Optimize button
        self.optimize_button = QPushButton("Optimize Portfolio")
        self.optimize_button.clicked.connect(self.optimize_portfolio)
        controls_layout.addWidget(self.optimize_button, 1, 0, 1, 2)
        
        # Progress bar
        self.opt_progress = QProgressBar()
        controls_layout.addWidget(self.opt_progress, 2, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Results
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Plot widget for efficient frontier
        self.opt_plot_widget = PlotWidget()
        self.opt_plot_widget.setBackground('w')
        results_splitter.addWidget(self.opt_plot_widget)
        
        # Results text
        self.opt_results_text = QTextEdit()
        self.opt_results_text.setReadOnly(True)
        results_splitter.addWidget(self.opt_results_text)
        
        layout.addWidget(results_splitter)
        
        self.tab_widget.addTab(tab, "Optimization")
    
    def create_visualization_tab(self):
        """Create visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Chart controls
        controls_group = QGroupBox("Chart Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        controls_layout.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Price Chart", "Returns Distribution", "Correlation Heatmap", "VaR Analysis"
        ])
        controls_layout.addWidget(self.chart_type_combo)
        
        self.plot_button = QPushButton("Generate Plot")
        self.plot_button.clicked.connect(self.generate_plot)
        controls_layout.addWidget(self.plot_button)
        
        layout.addWidget(controls_group)
        
        # Plot area
        self.plot_widget = PlotWidget()
        self.plot_widget.setBackground('w')
        layout.addWidget(self.plot_widget)
        
        # Matplotlib widget for correlation heatmap
        self.matplotlib_widget = QWidget()
        self.matplotlib_layout = QVBoxLayout(self.matplotlib_widget)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.matplotlib_layout.addWidget(self.canvas)
        layout.addWidget(self.matplotlib_widget)
        self.matplotlib_widget.hide()  # Initially hidden
        
        self.tab_widget.addTab(tab, "Visualization")
    
    def create_reports_tab(self):
        """Create reports tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Export controls
        export_group = QGroupBox("Export Reports")
        export_layout = QHBoxLayout(export_group)
        
        self.excel_button = QPushButton("Export to Excel")
        self.excel_button.clicked.connect(self.export_to_excel)
        export_layout.addWidget(self.excel_button)
        
        self.pdf_button = QPushButton("Export to PDF")
        self.pdf_button.clicked.connect(self.export_to_pdf)
        export_layout.addWidget(self.pdf_button)
        
        layout.addWidget(export_group)
        
        # Report preview
        preview_group = QGroupBox("Report Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        preview_layout.addWidget(self.report_preview)
        
        layout.addWidget(preview_group)
        
        self.tab_widget.addTab(tab, "📄 Reports")
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        dark_palette = QPalette()
        
        # Window colors
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        
        # Base colors
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        
        # Text colors
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        
        # Highlight colors
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        self.setPalette(dark_palette)
        
        # Set font
        font = QFont("Segoe UI", 9)
        self.setFont(font)
    
    def load_data(self):
        """Load portfolio data"""
        symbols = [s.strip().upper() for s in self.symbols_input.text().split(',')]
        start_date = self.start_date_input.text()
        end_date = self.end_date_input.text()
        
        if not symbols or not start_date or not end_date:
            QMessageBox.warning(self, "Warning", "Please fill in all fields")
            return
        
        # Disable button and reset progress
        self.load_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start data worker
        self.data_worker = DataWorker(symbols, start_date, end_date)
        self.data_worker.progress.connect(self.progress_bar.setValue)
        self.data_worker.finished.connect(self.on_data_loaded)
        self.data_worker.error.connect(self.on_data_error)
        self.data_worker.start()
    
    def on_data_loaded(self, data):
        """Handle loaded data"""
        self.portfolio_data = data
        self.load_button.setEnabled(True)
        
        # Update data table
        self.update_data_table()
        
        QMessageBox.information(self, "Success", f"Loaded data for {len(data)} symbols")
    
    def on_data_error(self, error):
        """Handle data loading error"""
        self.load_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to load data: {error}")
    
    def update_data_table(self):
        """Update data preview table"""
        if not self.portfolio_data:
            return
        
        # Get first symbol's data for preview
        first_symbol = list(self.portfolio_data.keys())[0]
        data = self.portfolio_data[first_symbol]
        
        # Show both first 10 and last 10 rows to demonstrate full range
        if len(data) > 20:
            # Show first 10 rows
            first_10 = data.head(10)
            # Show last 10 rows
            last_10 = data.tail(10)
            # Combine with separator
            preview_data = pd.concat([first_10, last_10])
            
            # Add a separator row to show the gap
            separator_row = pd.DataFrame(index=[pd.Timestamp('1900-01-01')], 
                                       columns=data.columns)
            separator_row.iloc[0] = ['...'] * len(data.columns)
            
            # Insert separator between first and last 10
            preview_data = pd.concat([first_10, separator_row, last_10])
        else:
            preview_data = data
        
        self.data_table.setRowCount(len(preview_data))
        self.data_table.setColumnCount(len(preview_data.columns) + 1)
        
        # Set headers
        headers = ['Date'] + list(preview_data.columns)
        self.data_table.setHorizontalHeaderLabels(headers)
        
        # Fill data
        for i, (date, row) in enumerate(preview_data.iterrows()):
            if date.year == 1900:  # Separator row
                self.data_table.setItem(i, 0, QTableWidgetItem("..."))
                for j in range(len(row)):
                    self.data_table.setItem(i, j + 1, QTableWidgetItem("..."))
            else:
                self.data_table.setItem(i, 0, QTableWidgetItem(str(date.date())))
                for j, value in enumerate(row):
                    if isinstance(value, str):  # Handle separator values
                        self.data_table.setItem(i, j + 1, QTableWidgetItem(value))
                    else:
                        self.data_table.setItem(i, j + 1, QTableWidgetItem(f"{value:.2f}"))
        
        # Add comprehensive summary info
        total_days = len(data)
        start_date = data.index[0].date()
        end_date = data.index[-1].date()
        years = (end_date - start_date).days / 365.25
        
        # Update window title with detailed info
        self.setWindowTitle(f"Modern Portfolio Risk Manager - {len(self.portfolio_data)} assets, {total_days} days ({years:.1f} years) [{start_date} to {end_date}]")
        
        # Also add a status message to the results text
        status_message = f"""
DATA LOADING STATUS
{'='*50}

Successfully loaded {len(self.portfolio_data)} assets
Date Range: {start_date} to {end_date}
Total Trading Days: {total_days:,}
Time Period: {years:.1f} years

Note: Table shows first 10 and last 10 rows for preview.
    All {total_days:,} days of data are loaded and available for analysis.
        """
        
        # If results_text exists, update it
        if hasattr(self, 'results_text'):
            self.results_text.setText(status_message)
    
    def calculate_var(self):
        """Calculate VaR and risk metrics"""
        if not self.portfolio_data:
            QMessageBox.warning(self, "Warning", "Please load portfolio data first")
            return
        
        confidence = self.confidence_spin.value()
        horizon = self.horizon_spin.value()
        
        try:
            # Calculate portfolio returns (equal weights)
            returns_data = {}
            for symbol, data in self.portfolio_data.items():
                returns = data['Close'].pct_change().dropna()
                returns_data[symbol] = returns
            
            returns_df = pd.DataFrame(returns_data)
            
            # Equal weights
            weights = np.array([1/len(returns_df.columns)] * len(returns_df.columns))
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Calculate VaR
            var_historical = np.percentile(portfolio_returns, (1 - confidence) * 100)
            var_parametric = stats.norm.ppf(1 - confidence, portfolio_returns.mean(), portfolio_returns.std())
            
            # Scale for time horizon
            var_historical_scaled = var_historical * np.sqrt(horizon)
            var_parametric_scaled = var_parametric * np.sqrt(horizon)
            
            # Expected Shortfall (CVaR)
            es_historical = portfolio_returns[portfolio_returns <= var_historical].mean()
            
            # Additional metrics
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            max_drawdown = self.calculate_max_drawdown(portfolio_returns)
            
            # Store results
            self.risk_metrics = {
                'portfolio_returns': portfolio_returns,
                'var_historical': var_historical_scaled,
                'var_parametric': var_parametric_scaled,
                'expected_shortfall': es_historical * np.sqrt(horizon),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'mean_return': portfolio_returns.mean() * 252
            }
            
            # Display results
            self.display_risk_results()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate VaR: {str(e)}")
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def display_risk_results(self):
        """Display risk analysis results"""
        if not self.risk_metrics:
            return
        
        results_text = f"""
PORTFOLIO RISK ANALYSIS RESULTS
{'='*50}

Value at Risk (VaR)
• Historical VaR ({self.confidence_spin.value():.0%}): {self.risk_metrics['var_historical']:.4f} ({self.risk_metrics['var_historical']*100:.2f}%)
• Parametric VaR ({self.confidence_spin.value():.0%}): {self.risk_metrics['var_parametric']:.4f} ({self.risk_metrics['var_parametric']*100:.2f}%)

Expected Shortfall (CVaR)
• Expected Shortfall: {self.risk_metrics['expected_shortfall']:.4f} ({self.risk_metrics['expected_shortfall']*100:.2f}%)

Portfolio Metrics
• Annual Return: {self.risk_metrics['mean_return']:.4f} ({self.risk_metrics['mean_return']*100:.2f}%)
• Annual Volatility: {self.risk_metrics['volatility']:.4f} ({self.risk_metrics['volatility']*100:.2f}%)
• Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.4f}
• Maximum Drawdown: {self.risk_metrics['max_drawdown']:.4f} ({self.risk_metrics['max_drawdown']*100:.2f}%)

Time Horizon: {self.horizon_spin.value()} days
Confidence Level: {self.confidence_spin.value():.0%}
        """
        
        self.results_text.setText(results_text)
    
    def run_monte_carlo(self):
        """Run Monte Carlo simulation"""
        if not self.portfolio_data:
            QMessageBox.warning(self, "Warning", "Please load portfolio data first")
            return
        
        # Prepare returns data
        returns_data = {}
        for symbol, data in self.portfolio_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        num_simulations = self.mc_simulations_spin.value()
        time_horizon = self.mc_horizon_spin.value()
        
        # Disable button and reset progress
        self.mc_button.setEnabled(False)
        self.mc_progress.setValue(0)
        
        # Start Monte Carlo worker
        self.mc_worker = MonteCarloWorker(returns_data, num_simulations, time_horizon)
        self.mc_worker.progress.connect(self.mc_progress.setValue)
        self.mc_worker.finished.connect(self.on_monte_carlo_finished)
        self.mc_worker.error.connect(self.on_monte_carlo_error)
        self.mc_worker.start()
    
    def on_monte_carlo_finished(self, results):
        """Handle Monte Carlo results"""
        self.monte_carlo_results = results
        self.mc_button.setEnabled(True)
        
        # Plot results
        self.plot_monte_carlo_results()
        
        # Display text results
        self.display_monte_carlo_results()
    
    def on_monte_carlo_error(self, error):
        """Handle Monte Carlo error"""
        self.mc_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Monte Carlo simulation failed: {error}")
    
    def plot_monte_carlo_results(self):
        """Plot Monte Carlo simulation results"""
        if not self.monte_carlo_results:
            return
        
        self.mc_plot_widget.clear()
        
        returns = self.monte_carlo_results['simulated_returns']
        
        # Create histogram
        y, x = np.histogram(returns, bins=50)
        self.mc_plot_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 100, 200, 150))
        
        # Add VaR lines
        var_95 = self.monte_carlo_results['var_95']
        var_99 = self.monte_carlo_results['var_99']
        
        self.mc_plot_widget.addLine(x=var_95, pen='red', label='VaR 95%')
        self.mc_plot_widget.addLine(x=var_99, pen='darkred', label='VaR 99%')
        
        self.mc_plot_widget.setLabel('left', 'Frequency')
        self.mc_plot_widget.setLabel('bottom', 'Portfolio Returns')
        self.mc_plot_widget.setTitle('Monte Carlo Simulation Results')
    
    def display_monte_carlo_results(self):
        """Display Monte Carlo text results"""
        if not self.monte_carlo_results:
            return
        
        results = self.monte_carlo_results
        
        results_text = f"""
MONTE CARLO SIMULATION RESULTS
{'='*50}

Simulation Parameters
• Number of Simulations: {self.mc_simulations_spin.value():,}
• Time Horizon: {self.mc_horizon_spin.value()} days

Portfolio Statistics
• Expected Return: {results['mean_return']:.4f} ({results['mean_return']*100:.2f}%)
• Standard Deviation: {results['std_return']:.4f} ({results['std_return']*100:.2f}%)
• Annual Expected Return: {results['portfolio_mean']:.4f} ({results['portfolio_mean']*100:.2f}%)
• Annual Volatility: {results['portfolio_std']:.4f} ({results['portfolio_std']*100:.2f}%)

Risk Metrics
• VaR (95%): {results['var_95']:.4f} ({results['var_95']*100:.2f}%)
• VaR (99%): {results['var_99']:.4f} ({results['var_99']*100:.2f}%)

Interpretation:
• There is a 5% chance of losing more than {abs(results['var_95']*100):.2f}%
• There is a 1% chance of losing more than {abs(results['var_99']*100):.2f}%
        """
        
        self.mc_results_text.setText(results_text)
    
    def optimize_portfolio(self):
        """Optimize portfolio"""
        if not self.portfolio_data:
            QMessageBox.warning(self, "Warning", "Please load portfolio data first")
            return
        
        # Prepare returns data
        returns_data = {}
        for symbol, data in self.portfolio_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        risk_free_rate = self.risk_free_spin.value()
        
        # Disable button and reset progress
        self.optimize_button.setEnabled(False)
        self.opt_progress.setValue(0)
        
        # Start optimization worker
        self.opt_worker = OptimizationWorker(returns_data, risk_free_rate)
        self.opt_worker.progress.connect(self.opt_progress.setValue)
        self.opt_worker.finished.connect(self.on_optimization_finished)
        self.opt_worker.error.connect(self.on_optimization_error)
        self.opt_worker.start()
    
    def on_optimization_finished(self, results):
        """Handle optimization results"""
        self.optimization_results = results
        self.optimize_button.setEnabled(True)
        
        # Plot efficient frontier
        self.plot_efficient_frontier()
        
        # Display text results
        self.display_optimization_results()
    
    def on_optimization_error(self, error):
        """Handle optimization error"""
        self.optimize_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Portfolio optimization failed: {error}")
    
    def plot_efficient_frontier(self):
        """Plot efficient frontier"""
        if not self.optimization_results:
            return
        
        self.opt_plot_widget.clear()
        
        # Plot efficient frontier
        frontier = self.optimization_results['efficient_frontier']
        if frontier:
            risks = [p['risk'] for p in frontier]
            returns = [p['return'] for p in frontier]
            
            self.opt_plot_widget.plot(risks, returns, pen='blue', name='Efficient Frontier')
        
        # Plot optimal portfolios
        max_sharpe = self.optimization_results['max_sharpe']
        min_var = self.optimization_results['min_variance']
        
        # Maximum Sharpe ratio point
        self.opt_plot_widget.plot([max_sharpe['risk']], [max_sharpe['return']], 
                                 pen=None, symbol='o', symbolBrush='red', symbolSize=10, name='Max Sharpe')
        
        # Minimum variance point
        self.opt_plot_widget.plot([min_var['risk']], [min_var['return']], 
                                 pen=None, symbol='s', symbolBrush='green', symbolSize=10, name='Min Variance')
        
        self.opt_plot_widget.setLabel('left', 'Expected Return')
        self.opt_plot_widget.setLabel('bottom', 'Risk (Standard Deviation)')
        self.opt_plot_widget.setTitle('Efficient Frontier')
        self.opt_plot_widget.addLegend()
    
    def display_optimization_results(self):
        """Display optimization text results"""
        if not self.optimization_results:
            return
        
        results = self.optimization_results
        max_sharpe = results['max_sharpe']
        min_var = results['min_variance']
        asset_names = results['asset_names']
        
        # Format weights
        max_sharpe_weights = "\n".join([f"  • {asset}: {weight:.2%}" for asset, weight in zip(asset_names, max_sharpe['weights'])])
        min_var_weights = "\n".join([f"  • {asset}: {weight:.2%}" for asset, weight in zip(asset_names, min_var['weights'])])
        
        results_text = f"""
PORTFOLIO OPTIMIZATION RESULTS
{'='*50}

Maximum Sharpe Ratio Portfolio
• Expected Return: {max_sharpe['return']:.4f} ({max_sharpe['return']*100:.2f}%)
• Risk (Std Dev): {max_sharpe['risk']:.4f} ({max_sharpe['risk']*100:.2f}%)
• Sharpe Ratio: {max_sharpe['sharpe']:.4f}
• Optimal Weights:
{max_sharpe_weights}

Minimum Variance Portfolio
• Expected Return: {min_var['return']:.4f} ({min_var['return']*100:.2f}%)
• Risk (Std Dev): {min_var['risk']:.4f} ({min_var['risk']*100:.2f}%)
• Optimal Weights:
{min_var_weights}

Analysis
• Risk-Free Rate: {self.risk_free_spin.value():.2%}
• Number of Assets: {len(asset_names)}
• Efficient Frontier Points: {len(results['efficient_frontier'])}

Recommendation:
• For maximum risk-adjusted returns, use the Maximum Sharpe Ratio portfolio
• For minimum risk, use the Minimum Variance portfolio
        """
        
        self.opt_results_text.setText(results_text)
    
    def generate_plot(self):
        """Generate selected plot"""
        if not self.portfolio_data:
            QMessageBox.warning(self, "Warning", "Please load portfolio data first")
            return
        
        chart_type = self.chart_type_combo.currentText()
        
        try:
            if chart_type == "Price Chart":
                self.plot_price_chart()
            elif chart_type == "Returns Distribution":
                self.plot_returns_distribution()
            elif chart_type == "Correlation Heatmap":
                self.plot_correlation_heatmap()
            elif chart_type == "VaR Analysis":
                self.plot_var_analysis()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate plot: {str(e)}")
    
    def plot_price_chart(self):
        """Plot price chart"""
        self.plot_widget.show()
        self.matplotlib_widget.hide()
        self.plot_widget.clear()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (symbol, data) in enumerate(self.portfolio_data.items()):
            dates = [d.timestamp() for d in data.index]
            prices = data['Close'].values
            
            color = colors[i % len(colors)]
            self.plot_widget.plot(dates, prices, pen=color, name=symbol)
        
        self.plot_widget.setLabel('left', 'Price ($)')
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.setTitle('Portfolio Price Chart')
        self.plot_widget.addLegend()
    
    def plot_returns_distribution(self):
        """Plot returns distribution"""
        self.plot_widget.show()
        self.matplotlib_widget.hide()
        
        if not self.risk_metrics or 'portfolio_returns' not in self.risk_metrics:
            QMessageBox.warning(self, "Warning", "Please calculate VaR first")
            return
        
        self.plot_widget.clear()
        returns = self.risk_metrics['portfolio_returns']
        
        # Create histogram
        y, x = np.histogram(returns, bins=50)
        self.plot_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        
        # Add VaR line
        if 'var_historical' in self.risk_metrics:
            var_line = self.risk_metrics['var_historical']
            self.plot_widget.addLine(x=var_line, pen='red', label='VaR')
        
        self.plot_widget.setLabel('left', 'Frequency')
        self.plot_widget.setLabel('bottom', 'Returns')
        self.plot_widget.setTitle('Portfolio Returns Distribution')
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap using matplotlib"""
        self.plot_widget.hide()
        self.matplotlib_widget.show()
        
        # Prepare returns data
        returns_data = {}
        for symbol, data in self.portfolio_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()
        
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_var_analysis(self):
        """Plot VaR analysis"""
        self.plot_widget.show()
        self.matplotlib_widget.hide()
        
        if not self.risk_metrics or 'portfolio_returns' not in self.risk_metrics:
            QMessageBox.warning(self, "Warning", "Please calculate VaR first")
            return
        
        self.plot_widget.clear()
        returns = self.risk_metrics['portfolio_returns']
        cumulative_returns = (1 + returns).cumprod()
        
        dates = [d.timestamp() for d in returns.index]
        
        self.plot_widget.plot(dates, cumulative_returns.values, pen='blue', name='Cumulative Returns')
        self.plot_widget.setLabel('left', 'Cumulative Returns')
        self.plot_widget.setLabel('bottom', 'Date')
        self.plot_widget.setTitle('Portfolio Cumulative Returns')
    
    def export_to_excel(self):
        """Export results to Excel"""
        if not self.portfolio_data:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Excel Report", "portfolio_report.xlsx", "Excel Files (*.xlsx)")
        
        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Export price data (remove timezone from datetime index)
                    for symbol, data in self.portfolio_data.items():
                        # Create a copy and remove timezone from index
                        data_copy = data.copy()
                        if hasattr(data_copy.index, 'tz') and data_copy.index.tz is not None:
                            data_copy.index = data_copy.index.tz_localize(None)
                        data_copy.to_excel(writer, sheet_name=f'{symbol}_Data')
                    
                    # Export risk metrics if available
                    if self.risk_metrics:
                        # Handle any datetime objects in risk metrics
                        metrics_dict = {}
                        for key, value in self.risk_metrics.items():
                            if isinstance(value, pd.Series) and hasattr(value.index, 'tz') and value.index.tz is not None:
                                # Convert timezone-aware series to timezone-naive
                                value_copy = value.copy()
                                value_copy.index = value_copy.index.tz_localize(None)
                                metrics_dict[key] = value_copy
                            elif hasattr(value, 'tz') and value.tz is not None:
                                # Handle individual datetime objects
                                metrics_dict[key] = value.tz_localize(None) if hasattr(value, 'tz_localize') else value
                            else:
                                metrics_dict[key] = value
                        
                        metrics_df = pd.DataFrame([metrics_dict])
                        metrics_df.to_excel(writer, sheet_name='Risk_Metrics')
                    
                    # Export Monte Carlo results if available
                    if self.monte_carlo_results:
                        mc_dict = {}
                        for key, value in self.monte_carlo_results.items():
                            if isinstance(value, pd.Series) and hasattr(value.index, 'tz') and value.index.tz is not None:
                                value_copy = value.copy()
                                value_copy.index = value_copy.index.tz_localize(None)
                                mc_dict[key] = value_copy
                            else:
                                mc_dict[key] = value
                        
                        mc_df = pd.DataFrame([mc_dict])
                        mc_df.to_excel(writer, sheet_name='Monte_Carlo')
                    
                    # Export optimization results if available
                    if self.optimization_results:
                        opt_summary = {
                            'Max_Sharpe_Return': self.optimization_results['max_sharpe']['return'],
                            'Max_Sharpe_Risk': self.optimization_results['max_sharpe']['risk'],
                            'Max_Sharpe_Ratio': self.optimization_results['max_sharpe']['sharpe'],
                            'Min_Var_Return': self.optimization_results['min_variance']['return'],
                            'Min_Var_Risk': self.optimization_results['min_variance']['risk']
                        }
                        opt_df = pd.DataFrame([opt_summary])
                        opt_df.to_excel(writer, sheet_name='Optimization')
            
                QMessageBox.information(self, "Success", f"Report exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def export_to_pdf(self):
        """Export results to PDF"""
        QMessageBox.information(self, "Info", "PDF export feature coming soon!")
    
    def closeEvent(self, event):
        """Handle application close"""
        if hasattr(self, 'conn'):
            self.conn.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ModernRiskManagerGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())