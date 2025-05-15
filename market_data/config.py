"""
Configuration settings for the financial data platform.
"""

# Market data settings
SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet (Google)
    "META",  # Meta (Facebook)
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
    "JPM",   # JPMorgan Chase
    "BAC",   # Bank of America
    "WMT",   # Walmart
]

# Data fetch intervals (in seconds)
PRICE_FETCH_INTERVAL = 60
OHLCV_FETCH_INTERVAL = 300  # 5 minutes

# Database settings
DATABASE_PATH = "finance.db"  # SQLite database path
