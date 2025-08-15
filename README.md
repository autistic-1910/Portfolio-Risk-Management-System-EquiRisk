# Portfolio Risk Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#license)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#installation)

A comprehensive desktop and web application for portfolio risk analysis, stress testing, and financial reporting built with Python and modern UI frameworks (PyQt5/Streamlit). This professional-grade tool provides advanced risk metrics, Monte Carlo simulations, GARCH volatility modeling, and detailed reporting capabilities.

## Quick Start

```bash
# Clone or download the project
# Navigate to the project director

# Install dependencies
pip install -r requirements.txt

# Run the GUI application
python modern_gui.py

# Or run the web interface
streamlit run streamlit_app.py
```

## Features Overview

### Data Management
- **Multi-Asset Support**: Handle portfolios with multiple stock symbols
- **Real-Time Data**: Automatic data retrieval from Yahoo Finance API
- **Persistent Storage**: SQLite database for historical data caching
- **Data Validation**: Robust error handling and data integrity checks
- **Flexible Date Ranges**: Customizable analysis periods

### Risk Analysis
- **Historical VaR**: Value at Risk using historical simulation method
- **Monte Carlo VaR**: 10,000-iteration Monte Carlo simulation for VaR calculation
- **GARCH Modeling**: Advanced volatility forecasting using GARCH(1,1) model
- **Expected Shortfall**: Conditional Value at Risk (CVaR) calculations
- **Multiple Confidence Levels**: Support for 90%, 95%, and 99% confidence intervals
- **Rolling Volatility**: 30-day rolling volatility analysis

### Stress Testing
- **Historical Scenarios**: Pre-configured stress tests for major market events:
  - 2008 Financial Crisis (Sep 2008 - Mar 2009)
  - COVID-19 Pandemic (Feb 2020 - Apr 2020)
  - Dot-com Bubble (Mar 2000 - Oct 2002)
- **Hypothetical Shocks**: Custom market and interest rate shock scenarios
- **Impact Visualization**: Clear charts showing portfolio performance under stress

### Visualization & Analytics
- **Interactive Charts**: Real-time matplotlib visualizations
- **Correlation Heatmaps**: Asset correlation analysis with color-coded matrices
- **Return Distribution**: Histogram plots with VaR thresholds
- **Cumulative Returns**: Portfolio performance tracking over time
- **Volatility Forecasting**: GARCH-based volatility predictions

### Reporting & Export
- **Comprehensive Excel Reports**: Multi-worksheet reports with:
  - Portfolio summary with current positions
  - Detailed calculation methodology
  - Risk metrics and VaR calculations
  - Complete historical data
  - Portfolio statistics (Sharpe ratio, skewness, kurtosis)
  - Stress test results
- **High-Resolution Chart Export**: 300 DPI PNG image exports
- **Dashboard State Saving**: JSON export for session persistence

## Installation

### Prerequisites
- **Python**: Version 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Internet Connection**: Required for downloading financial data

### Step-by-Step Installation

1. **Download the Project**
   ```bash
   # Ensure you have these files in your project directory:
   # - modern_gui.py (desktop application)
   # - streamlit_app.py (web interface)
   # - requirements.txt (dependencies)
   # - README.md (this file)
   # - portfolio_data.db (created automatically on first run)
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   If you encounter issues, try installing packages individually:
   ```bash
   pip install pandas numpy matplotlib yfinance scipy arch seaborn openpyxl sqlalchemy
   ```

3. **Verify Installation**
   ```bash
   python -c "import pandas, numpy, matplotlib, yfinance; print('All dependencies installed successfully!')"
   ```

4. **Launch Application**
   ```bash
   # For desktop GUI
   python modern_gui.py
   
   # For web interface
   streamlit run streamlit_app.py
   ```

   Note on desktop GUI: the current desktop app uses PyQt5. If your environment only has PyQt6 installed, install PyQt5 as well:
   ```bash
   pip install PyQt5
   ```

## Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| pandas | ≥2.0.0 | Data manipulation and analysis |
| numpy | ≥1.24.0 | Numerical computations |
| matplotlib | ≥3.7.0 | Plotting and visualization |
| yfinance | ≥0.2.0 | Yahoo Finance data download |
| scipy | ≥1.10.0 | Scientific computing and statistics |
| arch | ≥6.2.0 | GARCH volatility modeling |
| seaborn | ≥0.12.0 | Statistical data visualization |
| openpyxl | ≥3.1.0 | Excel file generation |
| sqlalchemy | ≥2.0.0 | Database operations |
| scikit-learn | ≥1.3.0 | Machine learning utilities |
| PyQt5 | ≥5.15.0 | Desktop GUI framework used by `modern_gui.py` |
| streamlit | ≥1.28.0 | Web interface framework |
| plotly | ≥5.17.0 | Interactive visualizations |
| dash | ≥2.14.0 | Web application framework |

*All dependencies have been updated to their latest stable versions for improved performance and security*

## Usage Guide

### Getting Started

1. **Launch the Application**
   - For desktop GUI: Run `python modern_gui.py`
   - For web interface: Run `streamlit run streamlit_app.py`
   - The application opens with multiple tabs for different functionalities

2. **Data Input Tab**
   - **Stock Symbols**: Enter comma-separated symbols (e.g., "AAPL,GOOGL,MSFT,TSLA,AMZN")
   - **Portfolio Weights**: Enter weights that sum to 1.0 (e.g., "0.2,0.2,0.2,0.2,0.2")
   - **Date Range**: Set analysis period (default: 2020-01-01 to current date)
   - **Load Data**: Click "Load Portfolio Data" to download and process data

3. **Risk Analysis Tab**
   - Select confidence levels (90%, 95%, 99%)
   - Choose analysis method:
     - **Historical VaR**: Uses actual historical returns
     - **Monte Carlo VaR**: Simulates 10,000 scenarios
     - **GARCH Model**: Advanced volatility forecasting
   - View results in four interactive charts:
     - Return distribution with VaR thresholds
     - Cumulative portfolio returns
     - 30-day rolling volatility
     - Asset correlation heatmap

4. **Stress Testing Tab**
   - **Historical Scenarios**: Select predefined crisis periods
   - **Hypothetical Shocks**: Configure custom market shocks
   - **Run Tests**: Execute stress tests and view impact analysis

5. **Reports & Export Tab**
   - **Excel Reports**: Generate comprehensive multi-worksheet reports
   - **Chart Export**: Save visualizations as high-resolution images
   - **Dashboard Save**: Export current state as JSON
   - **Summary Statistics**: View detailed portfolio metrics

### Advanced Features

#### GARCH Volatility Modeling
- Implements GARCH(1,1) specification for volatility clustering
- Provides 30-day volatility forecasts
- Accounts for heteroskedasticity in financial returns

#### Monte Carlo Simulation
- Uses multivariate normal distribution
- 10,000 simulation iterations for robust estimates
- Incorporates asset correlations in simulation

#### Stress Testing Methodology
- **Historical Scenarios**: Applies actual market conditions to current portfolio
- **Hypothetical Shocks**: Simulates user-defined market movements
- **Impact Calculation**: Shows percentage and dollar impact on portfolio value

## Technical Architecture

### Database Schema
```sql
-- Stock price data storage
CREATE TABLE stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume INTEGER,
    adj_close REAL,
    UNIQUE(symbol, date)
);

-- Portfolio allocation storage
CREATE TABLE portfolio_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    weight REAL NOT NULL,
    date_added DATE DEFAULT CURRENT_DATE
);
```

### Code Structure
- **modern_gui.py**: PyQt5-based desktop application with advanced GUI
- **streamlit_app.py**: Web-based interface using Streamlit framework
- **core/portfolio_manager.py**: Core portfolio management and risk calculations
- **reports/modern_reports.py**: Advanced reporting with interactive visualizations
- **Data Processing**: Robust data handling with yfinance integration
- **Risk Calculations**: Modular risk metric implementations
- **Visualization**: Modern charting with matplotlib and plotly
- **Export Functions**: Multi-format export capabilities

### Error Handling
- Comprehensive exception handling for data download failures
- Graceful degradation when symbols are unavailable
- User-friendly error messages and warnings
- Automatic data validation and weight normalization

## Output Examples

### Excel Report Structure
1. **Portfolio Summary**: Current holdings, weights, and basic metrics
2. **Calculation Methodology**: Detailed formulas and explanations
3. **Risk Metrics**: VaR and Expected Shortfall calculations
4. **Historical Data**: Complete price and return time series
5. **Portfolio Statistics**: Advanced metrics (Sharpe ratio, skewness, kurtosis)
6. **Stress Test Results**: Scenario analysis outcomes

### Chart Exports
- **Risk Analysis**: 4-panel chart with distribution, returns, volatility, and correlations
- **Stress Tests**: Bar charts and pie charts showing scenario impacts
- **High Resolution**: 300 DPI PNG format suitable for presentations

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No data could be loaded" | Invalid symbols or network issues | Check symbols and internet connection |
| "Weights must sum to 1.0" | Incorrect weight allocation | Ensure weights total exactly 1.0 |
| "ARCH library not available" | Missing dependency | Run `pip install arch` |
| Memory errors | Large datasets | Reduce portfolio size or date range |
| Slow performance | Network latency | Use shorter date ranges for initial testing |

### Debug Mode
For detailed error information, check the console output when running the application.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include error handling for new features
- Test with various portfolio configurations

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with relevant financial regulations when using this software for actual investment decisions.

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial professionals before making investment choices. The authors are not responsible for any financial losses resulting from the use of this software.

## Support

For technical support:
- Check the [troubleshooting section](#troubleshooting)
- Review console output for detailed error messages
- Ensure all dependencies are properly installed
- Verify internet connectivity for data downloads

## Version History

- **v1.0**: Initial release with basic risk analysis
- **v1.1**: Added GARCH modeling and stress testing
- **v1.2**: Enhanced reporting and export capabilities
- **v1.3**: Improved error handling and data validation
- **v1.4**: Added comprehensive Excel reporting with methodology
- **v2.0**: Major refactor with modernized dependencies and UI improvements

## Future Enhancements

- [ ] Enhance Streamlit interface; optional Dash support
- [ ] Real-time data streaming
- [ ] Additional risk models (Extreme Value Theory)
- [ ] Portfolio optimization features
- [ ] Backtesting capabilities
- [ ] API integration for institutional data providers

