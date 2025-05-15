"""
Advanced financial dashboard using Streamlit.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import logging
import sqlite3

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_data.data.analyzer import FinancialDataAnalyzer
from market_data.analytics.financial_metrics import FinancialMetrics
from market_data.analytics.anomaly_detection import MarketAnomalyDetector
from market_data.analytics.predictive_models import MarketPredictor
from market_data.storage.database_manager import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Advanced Financial Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'positive': '#4CAF50',
    'negative': '#E53935',
    'neutral': '#757575',
    'background': '#F5F5F5',
    'text': '#212121'
}

# Title and description
st.title("Financial Market Intelligence Platform")
st.markdown("*Real-time financial market data processing platform using streaming technologies and data engineering best practices*")

# Sidebar
st.sidebar.header("Dashboard Configuration")

# Database selection
db_files = [f for f in os.listdir('.') if f.endswith('.db')]
if not db_files:
    st.error("No database files found. Please run the market data producer first.")
    st.stop()

selected_db = st.sidebar.selectbox("Select Database", db_files, index=0)
db_path = selected_db

# Initialize database manager
db_manager = DatabaseManager(db_path)

# Get available symbols
try:
    latest_prices = db_manager.get_latest_prices()
    available_symbols = latest_prices['symbol'].unique().tolist()
except Exception as e:
    st.error(f"Error loading symbols from database: {e}")
    available_symbols = []

if not available_symbols:
    st.error("No symbols found in the database. Please run the market data producer first.")
    st.stop()

# Symbol selection
selected_symbols = st.sidebar.multiselect(
    "Select Stocks to Display",
    options=available_symbols,
    default=available_symbols[:4] if len(available_symbols) >= 4 else available_symbols
)

if not selected_symbols:
    st.warning("Please select at least one stock to display.")
    st.stop()
# Custom stock input
st.sidebar.markdown("---")
st.sidebar.subheader("Add Custom Stock")
custom_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, MSFT)").upper()

if custom_symbol:
    if custom_symbol in available_symbols:
        if custom_symbol not in selected_symbols:
            selected_symbols.append(custom_symbol)
            st.sidebar.success(f"Added {custom_symbol} to selected stocks!")
        else:
            st.sidebar.info(f"{custom_symbol} is already selected.")
    else:
        # Try to fetch data for this symbol
        try:
            # Import YahooFinance directly
            import yfinance as yf
            
            # Try to get quote from yfinance directly
            ticker = yf.Ticker(custom_symbol)
            info = ticker.info
            
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                # Get price information
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('previousClose', 0)
                
                # Calculate change
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close else 0
                
                # Create price data
                price_data = {
                    "symbol": custom_symbol,
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": info.get('regularMarketVolume', 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Get historical data directly
                hist = ticker.history(period="1mo")
                
                # Convert to list of dictionaries
                historical_data = []
                for idx, row in hist.iterrows():
                    historical_data.append({
                        "symbol": custom_symbol,
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": int(row['Volume']),
                        "period": "1d",
                        "timestamp": idx.to_pydatetime().isoformat()
                    })
                
                # Save to database directly without validation
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Add price data
                cursor.execute('''
                INSERT INTO stock_prices 
                (symbol, price, change, change_percent, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    price_data["symbol"],
                    price_data["price"],
                    price_data["change"],
                    price_data["change_percent"],
                    price_data["volume"],
                    price_data["timestamp"]
                ))
                
                # Add historical data
                for data in historical_data:
                    cursor.execute('''
                    INSERT INTO ohlcv 
                    (symbol, open, high, low, close, volume, period, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data["symbol"],
                        data["open"],
                        data["high"],
                        data["low"],
                        data["close"],
                        data["volume"],
                        data["period"],
                        data["timestamp"]
                    ))
                
                conn.commit()
                conn.close()
                
                # Add to available symbols
                available_symbols.append(custom_symbol)
                selected_symbols.append(custom_symbol)
                
                # Load data for the new symbol - FIXED
                # Instead of calling load_stock_data, directly create a DataFrame
                ohlcv_data = pd.DataFrame()
                try:
                    # Using direct DB query instead of load_stock_data
                    conn = sqlite3.connect(db_path)
                    query = f"SELECT * FROM ohlcv WHERE symbol = '{custom_symbol}'"
                    if start_date:
                        query += f" AND timestamp >= '{start_date.isoformat()}'"
                    if end_date:
                        query += f" AND timestamp <= '{end_date.isoformat()}'"
                    query += " ORDER BY timestamp ASC"
                    
                    ohlcv_data = pd.read_sql_query(query, conn)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in ohlcv_data.columns:
                        ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
                    
                    conn.close()
                except Exception as e:
                    st.sidebar.warning(f"Added stock but error loading data: {str(e)}")
                
                data_by_symbol[custom_symbol] = ohlcv_data
                
                # Force a page refresh to show the new data
                st.sidebar.success(f"Added {custom_symbol} to selected stocks! Refresh page to see data.")
                st.sidebar.button("Refresh Data")
            else:
                st.sidebar.error(f"Stock symbol '{custom_symbol}' not found")
        except Exception as e:
            st.sidebar.error(f"Error adding stock: {str(e)}")
# Date range selection
date_ranges = {
    "1 Day": timedelta(days=1),
    "1 Week": timedelta(days=7),
    "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90),
    "6 Months": timedelta(days=180),
    "1 Year": timedelta(days=365),
    "All": None
}

selected_range = st.sidebar.selectbox("Select Time Range", list(date_ranges.keys()), index=2)
date_range = date_ranges[selected_range]

# Calculate start and end dates
end_date = datetime.now()
start_date = end_date - date_range if date_range else None

# Technical indicator selection
st.sidebar.header("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=True)
show_macd = st.sidebar.checkbox("MACD", value=False)
show_volume = st.sidebar.checkbox("Volume", value=True)

# Analysis options
st.sidebar.header("Analysis Options")
show_anomalies = st.sidebar.checkbox("Detect Anomalies", value=True)
show_predictions = st.sidebar.checkbox("Price Predictions", value=True)
show_correlation = st.sidebar.checkbox("Correlation Analysis", value=False)
show_stats = st.sidebar.checkbox("Statistical Analysis", value=True)

# Main content
# Create tabs for different sections
tabs = st.tabs([
    "Market Overview", 
    "Stock Analysis", 
    "Technical Indicators", 
    "Anomaly Detection",
    "Price Prediction",
    "Data Explorer"
])

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(symbol, start_date=None, end_date=None):
    """Load stock data for a given symbol and date range."""
    try:
        if start_date:
            ohlcv_data = db_manager.get_historical_prices(symbol, start_date, end_date)
        else:
            ohlcv_data = db_manager.get_historical_prices(symbol)
        
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

# Load data for all selected symbols
data_by_symbol = {}
for symbol in selected_symbols:
    data_by_symbol[symbol] = load_stock_data(symbol, start_date, end_date)
# Add an auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)
    st.sidebar.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    # This creates an automatic refresh using JavaScript
    st.markdown(
        f"""
        <script>
            setTimeout(function(){{ window.location.reload(); }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )
# Tab 1: Market Overview
with tabs[0]:
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Performance")
        
        # Get latest prices for selected symbols
        latest_data = latest_prices[latest_prices['symbol'].isin(selected_symbols)]
        
        # Create performance chart
        fig = go.Figure()
        
        for symbol in selected_symbols:
            symbol_data = data_by_symbol[symbol]
            if not symbol_data.empty:
                # Normalize price to start at 100 for comparison
                first_price = symbol_data['close'].iloc[0]
                normalized_price = symbol_data['close'] / first_price * 100
                
                fig.add_trace(go.Scatter(
                    x=symbol_data['timestamp'],
                    y=normalized_price,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Relative Performance (Normalized to 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        # Show correlation matrix if selected
        if show_correlation and len(selected_symbols) > 1:
            st.subheader("Correlation Matrix")
            
            # Create DataFrame with close prices for all symbols
            price_df = pd.DataFrame()
            
            for symbol in selected_symbols:
                if not data_by_symbol[symbol].empty:
                    price_df[symbol] = data_by_symbol[symbol].set_index('timestamp')['close']
            
            # Calculate correlation matrix
            corr_matrix = price_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                color_continuous_midpoint=0
            )
            
            fig.update_layout(
                title="Price Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Summary")
        
        # Calculate market statistics
        avg_price = latest_data['price'].mean()
        avg_change = latest_data['change_percent'].mean()
        gainers = len(latest_data[latest_data['change'] > 0])
        losers = len(latest_data[latest_data['change'] < 0])
        total_volume = latest_data['volume'].sum()
        
        # Display metrics
        st.metric(
            label="Average Price",
            value=f"${avg_price:.2f}"
        )
        st.metric(
            label="Average Change",
            value=f"{avg_change:.2f}%",
            delta=f"{avg_change:.2f}%"
        )
        st.metric(
            label="Gainers vs. Losers",
            value=f"{gainers} ↑ / {losers} ↓"
        )
        st.metric(
            label="Total Volume",
            value=f"{total_volume:,}"
        )
        
        # Top gainers and losers
        st.subheader("Top Gainers")
        top_gainers = latest_data.sort_values('change_percent', ascending=False).head(3)
        for _, row in top_gainers.iterrows():
            st.metric(
                label=row['symbol'],
                value=f"${row['price']:.2f}",
                delta=f"{row['change_percent']:.2f}%"
            )
        
        st.subheader("Top Losers")
        top_losers = latest_data.sort_values('change_percent', ascending=True).head(3)
        for _, row in top_losers.iterrows():
            st.metric(
                label=row['symbol'],
                value=f"${row['price']:.2f}",
                delta=f"{row['change_percent']:.2f}%"
            )

# Tab 2: Stock Analysis
with tabs[1]:
    # Select a symbol for detailed analysis
    selected_symbol_analysis = st.selectbox(
        "Select a stock for detailed analysis",
        options=selected_symbols
    )
    
    # Load data for selected symbol
    stock_data = data_by_symbol[selected_symbol_analysis]
    
    if stock_data.empty:
        st.warning(f"No data available for {selected_symbol_analysis}")
    else:
        # Get latest price information
        latest_price = latest_prices[latest_prices['symbol'] == selected_symbol_analysis].iloc[0]
        
        # Create columns for layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${latest_price['price']:.2f}",
                delta=f"{latest_price['change']:.2f}"
            )
        
        with col2:
            st.metric(
                label="% Change",
                value=f"{latest_price['change_percent']:.2f}%",
                delta=f"{latest_price['change_percent']:.2f}%"
            )
        
        with col3:
            st.metric(
                label="Volume",
                value=f"{latest_price['volume']:,}"
            )
        
        # Price chart
        st.subheader(f"{selected_symbol_analysis} Price Chart")
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=stock_data['timestamp'],
                open=stock_data['open'],
                high=stock_data['high'],
                low=stock_data['low'],
                close=stock_data['close'],
                name="OHLC"
            )
        )
        # Add moving averages if selected
        if show_sma:
            # Calculate SMAs
            stock_data['sma_20'] = stock_data['close'].rolling(window=20).mean()
            stock_data['sma_50'] = stock_data['close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['sma_20'],
                    mode='lines',
                    name='SMA (20)',
                    line=dict(color='blue', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['sma_50'],
                    mode='lines',
                    name='SMA (50)',
                    line=dict(color='orange', width=1)
                )
            )
        
        # Add Bollinger Bands if selected
        if show_bollinger:
            # Calculate Bollinger Bands
            window = 20
            stock_data['middle_band'] = stock_data['close'].rolling(window=window).mean()
            stock_data['std'] = stock_data['close'].rolling(window=window).std()
            stock_data['upper_band'] = stock_data['middle_band'] + 2 * stock_data['std']
            stock_data['lower_band'] = stock_data['middle_band'] - 2 * stock_data['std']
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['upper_band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(0, 255, 0, 0.3)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data['timestamp'],
                    y=stock_data['lower_band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.1)'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_symbol_analysis} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart if selected
        if show_volume:
            vol_fig = go.Figure()
            
            vol_fig.add_trace(
                go.Bar(
                    x=stock_data['timestamp'],
                    y=stock_data['volume'],
                    name='Volume',
                    marker=dict(color='rgba(0, 128, 0, 0.7)')
                )
            )
            
            vol_fig.update_layout(
                title=f"{selected_symbol_analysis} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(vol_fig, use_container_width=True)
        
        # Statistical analysis if selected
        if show_stats:
            st.subheader("Statistical Analysis")
            
            # Calculate returns
            stock_data['daily_return'] = stock_data['close'].pct_change() * 100
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Summary statistics
                stats = stock_data['daily_return'].describe()
                
                metrics = {
                    "Average Daily Return": f"{stats['mean']:.2f}%",
                    "Return Volatility": f"{stats['std']:.2f}%",
                    "Minimum Return": f"{stats['min']:.2f}%",
                    "Maximum Return": f"{stats['max']:.2f}%"
                }
                
                for label, value in metrics.items():
                    st.metric(label=label, value=value)
            
            with col2:
                # Return distribution
                fig = px.histogram(
                    stock_data,
                    x='daily_return',
                    nbins=30,
                    title="Return Distribution",
                    labels={'daily_return': 'Daily Return (%)'}
                )
                
                # Add vertical line at mean
                fig.add_vline(
                    x=stats['mean'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean",
                    annotation_position="top"
                )
                
                fig.update_layout(height=300)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate cumulative returns
            stock_data['cumulative_return'] = (1 + stock_data['daily_return'] / 100).cumprod() * 100 - 100
            
            # Plot cumulative returns
            fig = px.line(
                stock_data,
                x='timestamp',
                y='cumulative_return',
                title=f"{selected_symbol_analysis} Cumulative Returns (%)",
                labels={'cumulative_return': 'Cumulative Return (%)', 'timestamp': 'Date'}
            )
            
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
# Tab 3: Technical Indicators
with tabs[2]:
    # Select a symbol for technical analysis
    selected_symbol_tech = st.selectbox(
        "Select a stock for technical analysis",
        options=selected_symbols,
        key="tech_analysis_symbol"
    )
    
    # Load data for selected symbol
    tech_data = data_by_symbol[selected_symbol_tech]
    
    if tech_data.empty:
        st.warning(f"No data available for {selected_symbol_tech}")
    else:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            if show_rsi:
                st.subheader("Relative Strength Index (RSI)")
                
                # Calculate RSI using Financial Metrics
                close_prices = tech_data['close'].values
                rsi_values = FinancialMetrics.calculate_rsi(close_prices)
                tech_data['rsi'] = rsi_values
                
                # Create RSI chart
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=tech_data['timestamp'],
                        y=tech_data['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    )
                )
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type="line",
                    x0=tech_data['timestamp'].iloc[0],
                    y0=70,
                    x1=tech_data['timestamp'].iloc[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=tech_data['timestamp'].iloc[0],
                    y0=30,
                    x1=tech_data['timestamp'].iloc[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash")
                )
                
                fig.update_layout(
                    title=f"{selected_symbol_tech} RSI (14)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # MACD
            if show_macd:
                st.subheader("Moving Average Convergence Divergence (MACD)")
                
                # Calculate MACD using Financial Metrics
                # MACD
                if show_macd:
                    st.subheader("Moving Average Convergence Divergence (MACD)")
                    
                    # Get close prices
                    close_prices = tech_data['close'].values
                    
                    # Check if we have enough data for MACD
                    if len(close_prices) >= 26:
                        # Calculate MACD using Financial Metrics
                        try:
                            macd_data = FinancialMetrics.calculate_macd(close_prices)
                            tech_data['macd'] = macd_data['macd']
                            tech_data['signal'] = macd_data['signal']
                            tech_data['histogram'] = macd_data['histogram']
                            
                            # Create MACD chart
                            fig = make_subplots(rows=1, cols=1)
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_data['timestamp'],
                                    y=tech_data['macd'],
                                    mode='lines',
                                    name='MACD',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_data['timestamp'],
                                    y=tech_data['signal'],
                                    mode='lines',
                                    name='Signal',
                                    line=dict(color='red', width=1)
                                )
                            )
                            
                            # Add histogram if available
                            if 'histogram' in tech_data.columns:
                                colors = ['green' if val >= 0 else 'red' for val in tech_data['histogram']]
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=tech_data['timestamp'],
                                        y=tech_data['histogram'],
                                        name='Histogram',
                                        marker=dict(color=colors)
                                    )
                                )
                            
                            fig.update_layout(
                                title=f"{selected_symbol_tech} MACD (12,26,9)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                height=300,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error calculating MACD: {str(e)}")
                            st.info("MACD calculation requires more data than is currently available.")
                    else:
                        # Not enough data
                        st.warning(f"MACD calculation requires at least 26 data points. Currently only have {len(close_prices)} points.")
                        
                        # Show a simplified version if possible
                        if len(close_prices) >= 10:
                            # Calculate simple moving averages as an alternative
                            short_ma = tech_data['close'].rolling(window=5).mean()
                            long_ma = tech_data['close'].rolling(window=10).mean()
                            tech_data['simple_macd'] = short_ma - long_ma
                            
                            # Create simple chart
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_data['timestamp'],
                                    y=tech_data['simple_macd'],
                                    mode='lines',
                                    name='Simple MACD (5,10)',
                                    line=dict(color='purple', width=2)
                                )
                            )
                            
                            fig.update_layout(
                                title=f"{selected_symbol_tech} Simplified MACD (5,10)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("Showing simplified MACD using shorter periods (5,10) instead of standard (12,26,9)")
                

        
        # Bollinger Bands
        if show_bollinger:
            st.subheader("Bollinger Bands")
            
            # Calculate Bollinger Bands using Financial Metrics
            close_prices = tech_data['close'].values
            bb_data = FinancialMetrics.calculate_bollinger_bands(close_prices)
            
            tech_data['middle_band'] = bb_data['middle']
            tech_data['upper_band'] = bb_data['upper']
            tech_data['lower_band'] = bb_data['lower']
            
            # Create Bollinger Bands chart
            fig = go.Figure()
            
            # Add price
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=1)
                )
            )
            
            # Add bands
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['upper_band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(0, 255, 0, 0.3)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['middle_band'],
                    mode='lines',
                    name='Middle Band (SMA 20)',
                    line=dict(color='rgba(0, 0, 255, 0.3)', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data['lower_band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.1)'
                )
            )
            
            fig.update_layout(
                title=f"{selected_symbol_tech} Bollinger Bands (20,2)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Technical analysis summary
        st.subheader("Technical Analysis Summary")
        
        # Create a dataframe with the latest values
        latest_tech = tech_data.iloc[-1]
        
        # Create columns for layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Close Price",
                value=f"${latest_tech['close']:.2f}"
            )
            
            if 'sma_20' in tech_data.columns and 'sma_50' in tech_data.columns:
                # Calculate SMAs if not already calculated
                if 'sma_20' not in tech_data.columns:
                    tech_data['sma_20'] = tech_data['close'].rolling(window=20).mean()
                if 'sma_50' not in tech_data.columns:
                    tech_data['sma_50'] = tech_data['close'].rolling(window=50).mean()
                    
                latest_tech = tech_data.iloc[-1]
                
                st.metric(
                    label="SMA (20)",
                    value=f"${latest_tech['sma_20']:.2f}",
                    delta=f"{(latest_tech['close'] - latest_tech['sma_20']):.2f}"
                )
                
                st.metric(
                    label="SMA (50)",
                    value=f"${latest_tech['sma_50']:.2f}",
                    delta=f"{(latest_tech['close'] - latest_tech['sma_50']):.2f}"
                )
        
        with col2:
            if 'rsi' in tech_data.columns:
                rsi_value = latest_tech['rsi']
                rsi_color = (
                    "red" if rsi_value > 70 else 
                    "green" if rsi_value < 30 else 
                    "black"
                )
                
                st.markdown(f"<h3 style='color:{rsi_color}'>RSI (14): {rsi_value:.2f}</h3>", unsafe_allow_html=True)
                
                rsi_signal = (
                    "Overbought" if rsi_value > 70 else
                    "Oversold" if rsi_value < 30 else
                    "Neutral"
                )
                
                st.markdown(f"**Signal:** {rsi_signal}")
            
            if 'macd' in tech_data.columns and 'signal' in tech_data.columns:
                macd_value = latest_tech['macd']
                signal_value = latest_tech['signal']
                macd_color = "green" if macd_value > signal_value else "red"
                
                st.markdown(f"<h3 style='color:{macd_color}'>MACD: {macd_value:.4f}</h3>", unsafe_allow_html=True)
                
                macd_signal = (
                    "Bullish" if macd_value > signal_value else
                    "Bearish"
                )
                
                st.markdown(f"**Signal:** {macd_signal}")
        
        with col3:
            if 'upper_band' in tech_data.columns and 'lower_band' in tech_data.columns:
                upper_band = latest_tech['upper_band']
                lower_band = latest_tech['lower_band']
                close_price = latest_tech['close']
                
                bb_position = (close_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
                
                bb_signal = (
                    "Overbought" if close_price > upper_band else
                    "Oversold" if close_price < lower_band else
                    "Neutral"
                )
                
                st.markdown(f"<h3>BB Position: {bb_position:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"**Signal:** {bb_signal}")
# Tab 4: Anomaly Detection
with tabs[3]:
    if show_anomalies:
        st.subheader("Market Anomaly Detection")
        
        # Select symbol for anomaly detection
        anomaly_symbol = st.selectbox(
            "Select a stock for anomaly detection",
            options=selected_symbols,
            key="anomaly_symbol"
        )
        
        # Load data for selected symbol
        anomaly_data = data_by_symbol[anomaly_symbol]
        
        if anomaly_data.empty:
            st.warning(f"No data available for {anomaly_symbol}")
        else:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Parameters for anomaly detection
            with col1:
                st.subheader("Detection Parameters")
                
                anomaly_window = st.slider(
                    "Window Size",
                    min_value=5,
                    max_value=60,
                    value=20,
                    step=5,
                    help="Number of periods to use for baseline calculation"
                )
                
                anomaly_threshold = st.slider(
                    "Anomaly Threshold",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    help="Threshold in standard deviations for anomaly detection"
                )
                
                anomaly_types = st.multiselect(
                    "Anomaly Types to Detect",
                    options=["Price", "Volume", "Volatility", "Momentum"],
                    default=["Price", "Volume"]
                )
            
            with col2:
                st.subheader("Detection Results")
                
                # Initialize anomaly detector
                detector = MarketAnomalyDetector()
                
                # Detect price anomalies
                total_anomalies = 0
                
                if "Price" in anomaly_types:
                    price_anomalies = detector.detect_price_anomalies(
                        anomaly_data['close'].values,
                        window=anomaly_window,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['price_anomaly'] = price_anomalies
                    price_anomaly_count = np.sum(price_anomalies)
                    total_anomalies += price_anomaly_count
                    
                    st.metric(
                        label="Price Anomalies",
                        value=price_anomaly_count
                    )
                
                # Detect volume anomalies
                if "Volume" in anomaly_types:
                    volume_anomalies = detector.detect_volume_anomalies(
                        anomaly_data['volume'].values,
                        window=anomaly_window,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['volume_anomaly'] = volume_anomalies
                    volume_anomaly_count = np.sum(volume_anomalies)
                    total_anomalies += volume_anomaly_count
                    
                    st.metric(
                        label="Volume Anomalies",
                        value=volume_anomaly_count
                    )
                
                # Detect volatility regime changes
                if "Volatility" in anomaly_types:
                    # Calculate returns
                    returns = np.diff(anomaly_data['close'].values) / anomaly_data['close'].values[:-1]
                    # Add a 0 at the beginning to maintain the same array length
                    returns = np.insert(returns, 0, 0)
                    
                    volatility_regimes = detector.detect_volatility_regime_changes(
                        returns,
                        window=anomaly_window,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['high_volatility'] = volatility_regimes['high_volatility']
                    anomaly_data['low_volatility'] = volatility_regimes['low_volatility']
                    
                    vol_anomaly_count = np.sum(volatility_regimes['high_volatility']) + np.sum(volatility_regimes['low_volatility'])
                    total_anomalies += vol_anomaly_count
                    
                    st.metric(
                        label="Volatility Regime Changes",
                        value=vol_anomaly_count
                    )
                
                # Detect momentum anomalies
                if "Momentum" in anomaly_types:
                    momentum_anomalies = detector.detect_momentum_anomalies(
                        anomaly_data['close'].values,
                        short_window=5,
                        long_window=20,
                        threshold=anomaly_threshold
                    )
                    
                    anomaly_data['positive_momentum'] = momentum_anomalies['positive_momentum']
                    anomaly_data['negative_momentum'] = momentum_anomalies['negative_momentum']
                    
                    momentum_anomaly_count = np.sum(momentum_anomalies['positive_momentum']) + np.sum(momentum_anomalies['negative_momentum'])
                    total_anomalies += momentum_anomaly_count
                    
                    st.metric(
                        label="Momentum Anomalies",
                        value=momentum_anomaly_count
                    )
                
                st.metric(
                    label="Total Anomalies",
                    value=total_anomalies
                )
            
            # Visualization of anomalies
            st.subheader("Anomaly Visualization")
            
            # Create price chart with anomalies
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=1)
                )
            )
            
            # Add price anomalies if detected
            if "Price" in anomaly_types:
                # Get indices of price anomalies
                price_anomaly_indices = np.where(anomaly_data['price_anomaly'])[0]
                
                if len(price_anomaly_indices) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_data.iloc[price_anomaly_indices]['timestamp'],
                            y=anomaly_data.iloc[price_anomaly_indices]['close'],
                            mode='markers',
                            name='Price Anomalies',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='circle',
                                line=dict(color='black', width=1)
                            )
                        )
                    )
            
            # Add momentum anomalies if detected
            if "Momentum" in anomaly_types and 'positive_momentum' in anomaly_data.columns:
                # Get indices of positive momentum anomalies
                pos_momentum_indices = np.where(anomaly_data['positive_momentum'])[0]
                
                if len(pos_momentum_indices) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_data.iloc[pos_momentum_indices]['timestamp'],
                            y=anomaly_data.iloc[pos_momentum_indices]['close'],
                            mode='markers',
                            name='Positive Momentum',
                            marker=dict(
                                color='green',
                                size=10,
                                symbol='triangle-up',
                                line=dict(color='black', width=1)
                            )
                        )
                    )
                
                # Get indices of negative momentum anomalies
                neg_momentum_indices = np.where(anomaly_data['negative_momentum'])[0]
                
                if len(neg_momentum_indices) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_data.iloc[neg_momentum_indices]['timestamp'],
                            y=anomaly_data.iloc[neg_momentum_indices]['close'],
                            mode='markers',
                            name='Negative Momentum',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='triangle-down',
                                line=dict(color='black', width=1)
                            )
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title=f"{anomaly_symbol} Price with Anomalies",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume anomalies chart
            if "Volume" in anomaly_types:
                vol_fig = go.Figure()
                
                # Add volume bars
                vol_fig.add_trace(
                    go.Bar(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['volume'],
                        name='Volume',
                        marker=dict(color='rgba(0, 128, 0, 0.5)')
                    )
                )
                
                # Add volume anomalies
                volume_anomaly_indices = np.where(anomaly_data['volume_anomaly'])[0]
                
                if len(volume_anomaly_indices) > 0:
                    vol_fig.add_trace(
                        go.Bar(
                            x=anomaly_data.iloc[volume_anomaly_indices]['timestamp'],
                            y=anomaly_data.iloc[volume_anomaly_indices]['volume'],
                            name='Volume Anomalies',
                            marker=dict(color='rgba(255, 0, 0, 0.8)')
                        )
                    )
                
                # Update layout
                vol_fig.update_layout(
                    title=f"{anomaly_symbol} Volume with Anomalies",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("Enable 'Detect Anomalies' in the sidebar to use this feature.")
# Tab 5: Price Prediction
with tabs[4]:
    if show_predictions:
        st.subheader("Price Prediction Models")
        
        # Select symbol for price prediction
        prediction_symbol = st.selectbox(
            "Select a stock for price prediction",
            options=selected_symbols,
            key="prediction_symbol"
        )
        
        # Load data for selected symbol
        prediction_data = data_by_symbol[prediction_symbol]
        
        if prediction_data.empty:
            st.warning(f"No data available for {prediction_symbol}")
        else:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Parameters for prediction
            with col1:
                st.subheader("Model Parameters")
                
                pred_horizon = st.slider(
                    "Prediction Horizon (days)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1,
                    help="Number of days to predict into the future"
                )
                
                pred_features = st.multiselect(
                    "Features to Use",
                    options=["Price", "Volume", "Technical Indicators"],
                    default=["Price", "Technical Indicators"]
                )
                
                test_size = st.slider(
                    "Test Set Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Percentage of data to use for testing"
                ) / 100
            
            with col2:
                st.subheader("Run Prediction")
                
                run_prediction = st.button("Run Price Prediction Model")
                
                if run_prediction:
                    with st.spinner("Training prediction models..."):
                        # Initialize predictor
                        predictor = MarketPredictor()
                        
                        # Prepare data
                        # Get close prices
                        prices = prediction_data['close']
                        
                        # Create additional features if selected
                        additional_features = None
                        
                        if "Volume" in pred_features or "Technical Indicators" in pred_features:
                            additional_features = pd.DataFrame(index=prediction_data.index)
                            
                            if "Volume" in pred_features:
                                # Add volume features
                                additional_features['volume'] = prediction_data['volume']
                                additional_features['volume_change'] = prediction_data['volume'].pct_change()
                                additional_features['volume_ma5'] = prediction_data['volume'].rolling(window=5).mean()
                            
                            if "Technical Indicators" in pred_features:
                                # Add technical indicators
                                # Calculate RSI
                                additional_features['rsi'] = FinancialMetrics.calculate_rsi(prediction_data['close'].values)
                                
                                # Calculate SMAs
                                additional_features['sma_5'] = prediction_data['close'].rolling(window=5).mean()
                                additional_features['sma_20'] = prediction_data['close'].rolling(window=20).mean()
                                
                                # Calculate volatility
                                returns = prediction_data['close'].pct_change().values
                                additional_features['volatility'] = FinancialMetrics.calculate_volatility(returns, window=20, annualize=False)
                        
                        # Run prediction model
                        results = predictor.predict_price_movement(
                            prices,
                            additional_features=additional_features,
                            n_lags=pred_horizon,
                            test_size=test_size
                        )
                        
                        if 'error' in results:
                            st.error(f"Error in prediction: {results['error']}")
                        else:
                            st.success("Prediction model trained successfully!")
                            
                            # Show best model
                            best_model = results['best_model']
                            st.info(f"Best model: {best_model}")
                            
                            # Show model performance
                            model_results = results['results'][best_model]
                            
                            metrics = {
                                "Train RMSE": f"${model_results['train_rmse']:.3f}",
                                "Test RMSE": f"${model_results['test_rmse']:.3f}",
                                "Train MAE": f"${model_results['train_mae']:.3f}",
                                "Test MAE": f"${model_results['test_mae']:.3f}",
                                "Train R²": f"{model_results['train_r2']:.3f}",
                                "Test R²": f"{model_results['test_r2']:.3f}"
                            }
                            
                            # Create metrics display
                            metric_cols = st.columns(3)
                            
                            for i, (label, value) in enumerate(metrics.items()):
                                col_idx = i % 3
                                metric_cols[col_idx].metric(label=label, value=value)
                            
                            # Show future prediction
                            if results['future_prediction'] is not None:
                                future_price = results['future_prediction']
                                latest_price = prediction_data['close'].iloc[-1]
                                price_change = future_price - latest_price
                                pct_change = price_change / latest_price * 100
                                
                                st.subheader("Price Prediction")
                                st.metric(
                                    label=f"Predicted Price in {pred_horizon} days",
                                    value=f"${future_price:.2f}",
                                    delta=f"{pct_change:.2f}%"
                                )
                            
                            # Show prediction chart
                            st.subheader("Model Performance")
                            
                            # Create plot of actual vs predicted values
                            fig = go.Figure()
                            
                            # Get actual and predicted values
                            test_actual = model_results['test_actual']
                            test_pred = model_results['test_predictions']
                            
                            # Get the dates for the test set (last x% of the data)
                            test_size_points = int(len(prediction_data) * test_size)
                            test_dates = prediction_data['timestamp'].iloc[-test_size_points:].reset_index(drop=True)
                            
                            # Add actual prices
                            fig.add_trace(
                                go.Scatter(
                                    x=test_dates,
                                    y=test_actual,
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # Add predicted prices
                            fig.add_trace(
                                go.Scatter(
                                    x=test_dates,
                                    y=test_pred,
                                    mode='lines',
                                    name='Predicted',
                                    line=dict(color='red', width=2, dash='dash')
                                )
                            )
                            
                            # Add future prediction point if available
                            if results['future_prediction'] is not None:
                                future_date = prediction_data['timestamp'].iloc[-1] + pd.Timedelta(days=pred_horizon)
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[future_date],
                                        y=[future_price],
                                        mode='markers',
                                        name='Future Prediction',
                                        marker=dict(
                                            color='green',
                                            size=10,
                                            symbol='star',
                                            line=dict(color='black', width=1)
                                        )
                                    )
                                )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"{prediction_symbol} Price Prediction",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=400,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Click 'Run Price Prediction Model' to train and evaluate prediction models.")
    else:
        st.info("Enable 'Price Predictions' in the sidebar to use this feature.")
# Tab 6: Data Explorer
with tabs[5]:
    st.subheader("Financial Data Explorer")
    
    # Select data type
    data_type = st.radio(
        "Select Data Type",
        options=["Price Data", "OHLCV Data", "Statistics"],
        horizontal=True
    )
    
    # Select symbol
    explorer_symbol = st.selectbox(
        "Select Symbol",
        options=selected_symbols,
        key="explorer_symbol"
    )
    
    # Load data for selected symbol
    explorer_data = data_by_symbol[explorer_symbol]
    
    if explorer_data.empty:
        st.warning(f"No data available for {explorer_symbol}")
    else:
        if data_type == "Price Data":
            # Get price data from database
            price_data = db_manager.get_latest_prices([explorer_symbol])
            
            st.subheader(f"Latest Price Data for {explorer_symbol}")
            st.dataframe(price_data)
            
            # Download button
            csv = price_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Price Data as CSV",
                data=csv,
                file_name=f"{explorer_symbol}_prices.csv",
                mime="text/csv"
            )
        
        elif data_type == "OHLCV Data":
            st.subheader(f"OHLCV Data for {explorer_symbol}")
            
            # Date filter
            col1, col2 = st.columns(2)
            
            with col1:
                start_filter = st.date_input(
                    "Start Date",
                    value=explorer_data['timestamp'].min().date() if not explorer_data.empty else None
                )
            
            with col2:
                end_filter = st.date_input(
                    "End Date",
                    value=explorer_data['timestamp'].max().date() if not explorer_data.empty else None
                )
            
            # Filter data
            filtered_data = explorer_data[
                (explorer_data['timestamp'].dt.date >= start_filter) &
                (explorer_data['timestamp'].dt.date <= end_filter)
            ]
            
            st.dataframe(filtered_data.sort_values('timestamp', ascending=False))
            
            # Download button
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download OHLCV Data as CSV",
                data=csv,
                file_name=f"{explorer_symbol}_ohlcv.csv",
                mime="text/csv"
            )
        
        elif data_type == "Statistics":
            st.subheader(f"Statistical Analysis for {explorer_symbol}")
            
            # Calculate returns
            explorer_data['daily_return'] = explorer_data['close'].pct_change() * 100
            
            # Get monthly returns
            explorer_data['year_month'] = explorer_data['timestamp'].dt.to_period('M')
            monthly_returns = explorer_data.groupby('year_month')['daily_return'].agg(['mean', 'std', 'min', 'max']).reset_index()
            monthly_returns['year_month'] = monthly_returns['year_month'].astype(str)
            
            # Get yearly returns
            explorer_data['year'] = explorer_data['timestamp'].dt.year
            yearly_returns = explorer_data.groupby('year')['daily_return'].agg(['mean', 'std', 'min', 'max']).reset_index()
            

            # Create tabs for different statistics
            stat_tabs = st.tabs(["Summary", "Monthly", "Yearly", "Distribution"])
            
            with stat_tabs[0]:
                st.subheader("Summary Statistics")
                
                # Calculate statistics
                summary = explorer_data['daily_return'].describe()
                
                # Calculate additional metrics
                total_trading_days = len(explorer_data)
                positive_days = (explorer_data['daily_return'] > 0).sum()
                negative_days = (explorer_data['daily_return'] < 0).sum()
                positive_pct = positive_days / total_trading_days * 100
                
                # Calculate annualized return and volatility
                ann_return = explorer_data['daily_return'].mean() * 252
                ann_volatility = explorer_data['daily_return'].std() * np.sqrt(252)
                sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Annualized Return", value=f"{ann_return:.2f}%")
                    st.metric(label="Daily Avg Return", value=f"{summary['mean']:.2f}%")
                    st.metric(label="Total Trading Days", value=f"{total_trading_days}")
                
                with col2:
                    st.metric(label="Annualized Volatility", value=f"{ann_volatility:.2f}%")
                    st.metric(label="Daily Volatility", value=f"{summary['std']:.2f}%")
                    st.metric(label="Positive Days", value=f"{positive_days} ({positive_pct:.1f}%)")
                
                with col3:
                    st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
                    st.metric(label="Max Daily Gain", value=f"{summary['max']:.2f}%")
                    st.metric(label="Max Daily Loss", value=f"{summary['min']:.2f}%")
            
            with stat_tabs[1]:
                st.subheader("Monthly Returns")
                
                # Display monthly returns
                st.dataframe(monthly_returns)
                
                # Create heatmap of monthly returns
                monthly_returns['year'] = monthly_returns['year_month'].str.split('-').str[0]
                monthly_returns['month'] = monthly_returns['year_month'].str.split('-').str[1]
                
                # Create pivot table
                pivot_df = monthly_returns.pivot(index='year', columns='month', values='mean')
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df,
                    text_auto=".2f",
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    color_continuous_midpoint=0,
                    labels=dict(x="Month", y="Year", color="Return (%)")
                )
                
                fig.update_layout(
                    title="Monthly Returns Heatmap (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_tabs[2]:
                st.subheader("Yearly Returns")
                
                # Display yearly returns
                st.dataframe(yearly_returns)
                
                # Create bar chart of yearly returns
                fig = px.bar(
                    yearly_returns,
                    x='year',
                    y='mean',
                    error_y='std',
                    labels=dict(x="Year", y="Return (%)", mean="Average Return"),
                    title=f"{explorer_symbol} Yearly Returns",
                    color='mean',
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    color_continuous_midpoint=0
                )
                
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with stat_tabs[3]:
                st.subheader("Return Distribution")
                
                # Create histogram of returns
                fig = px.histogram(
                    explorer_data,
                    x='daily_return',
                    nbins=50,
                    labels=dict(x="Daily Return (%)", y="Frequency"),
                    title=f"{explorer_symbol} Return Distribution"
                )
                
                # Add a vertical line at the mean
                fig.add_vline(
                    x=explorer_data['daily_return'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean",
                    annotation_position="top"
                )
                
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Clean returns data
                returns = explorer_data['daily_return'].dropna()

                # Check if we have sufficient data
                if len(returns) > 3 and returns.var() > 0:
                    # Calculate and display skewness and kurtosis
                    skew = returns.skew()
                    kurt = returns.kurtosis()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Skewness", value=f"{skew:.4f}")
                        skew_interpretation = (
                            "Positive (right-skewed)" if skew > 0 else
                            "Negative (left-skewed)" if skew < 0 else
                            "Zero (symmetric)"
                        )
                        st.markdown(f"**Interpretation:** {skew_interpretation}")
                    
                    with col2:
                        st.metric(label="Kurtosis", value=f"{kurt:.4f}")
                        kurt_interpretation = (
                            "Leptokurtic (heavy tails)" if kurt > 0 else
                            "Platykurtic (thin tails)" if kurt < 0 else
                            "Mesokurtic (normal)"
                        )
                        st.markdown(f"**Interpretation:** {kurt_interpretation}")
                else:
                    st.warning("Insufficient data for calculating skewness and kurtosis. Need at least 4 valid data points with variation.")
                # QQ plot
                import scipy.stats as stats
                
                # Calculate QQ plot data
                returns = explorer_data['daily_return'].dropna()
                qq = stats.probplot(returns, dist="norm")
                
                # Extract data
                x = np.array([point[0] for point in qq[0]])
                y = np.array([point[1] for point in qq[0]])
                
                # Create QQ plot
                fig = go.Figure()
                
                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name='Returns',
                        marker=dict(color='blue')
                    )
                )
                
                # Add the line representing normal distribution
                slope = qq[1][0]
                intercept = qq[1][1]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=slope * x + intercept,
                        mode='lines',
                        name='Normal',
                        line=dict(color='red')
                    )
                )
                
                fig.update_layout(
                    title="Normal Q-Q Plot",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Normality test
                st.subheader("Normality Test (Shapiro-Wilk)")

                # Check if we have sufficient data (Shapiro-Wilk requires at least 3 data points)
                if len(returns) >= 3 and returns.var() > 0:
                    try:
                        stat, p_value = stats.shapiro(returns)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(label="Test Statistic", value=f"{stat:.4f}")
                        
                        with col2:
                            st.metric(label="p-value", value=f"{p_value:.8f}")
                        
                        # Interpret the result
                        if p_value < 0.05:
                            st.markdown("**Conclusion:** Reject the null hypothesis. Returns are **not normally distributed**.")
                        else:
                            st.markdown("**Conclusion:** Fail to reject the null hypothesis. Returns may follow a normal distribution.")
                    except Exception as e:
                        st.warning(f"Could not perform Shapiro-Wilk test: {str(e)}")
                else:
                    st.warning("Insufficient data for Shapiro-Wilk test. Need at least 3 valid data points with variation.")