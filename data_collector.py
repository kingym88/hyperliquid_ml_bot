import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

class DataCollector:
    """Collects and manages cryptocurrency price data"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Define symbols
        self.symbols = {
            'BTC': 'BTC/USDT',
            'HBAR': 'HBAR/USDT',
            'XRP': 'XRP/USDT',
            'XLM': 'XLM/USDT'
        }
        
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_historical_data(self, symbol, timeframe='1m', 
                            since_date=None, limit=1000):
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m' for minute)
            since_date: Start date (datetime object)
            limit: Number of candles per request
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if since_date:
                since = int(since_date.timestamp() * 1000)
            else:
                since = None
            
            all_candles = []
            
            while True:
                candles = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Check if we got all data
                if len(candles) < limit:
                    break
                
                # Update since for next batch
                since = candles[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_all_symbols(self, start_date=None):
        """
        Collect data for all symbols
        
        Args:
            start_date: Start date string 'YYYY-MM-DD'
        
        Returns:
            Dictionary of DataFrames
        """
        if start_date:
            since = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            since = datetime.now() - timedelta(days=30)
        
        data = {}
        
        for coin, symbol in self.symbols.items():
            print(f"Fetching {coin} data...")
            df = self.fetch_historical_data(symbol, since_date=since)
            
            if not df.empty:
                data[coin] = df
                # Save to CSV
                filepath = os.path.join(self.data_dir, f'{coin}_1m.csv')
                df.to_csv(filepath)
                print(f"Saved {len(df)} candles for {coin}")
            
            time.sleep(1)  # Rate limiting
        
        return data
    
    def load_saved_data(self, coin):
        """Load previously saved data"""
        filepath = os.path.join(self.data_dir, f'{coin}_1m.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        return pd.DataFrame()
    
    def get_latest_price(self, symbol):
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def update_data(self):
        """Update data with latest candles"""
        for coin, symbol in self.symbols.items():
            df_old = self.load_saved_data(coin)
            
            if df_old.empty:
                continue
            
            # Get last timestamp
            last_timestamp = df_old.index[-1]
            since_date = last_timestamp + timedelta(minutes=1)
            
            # Fetch new data
            df_new = self.fetch_historical_data(
                symbol, 
                since_date=since_date
            )
            
            if not df_new.empty:
                # Combine and save
                df_combined = pd.concat([df_old, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                
                filepath = os.path.join(self.data_dir, f'{coin}_1m.csv')
                df_combined.to_csv(filepath)
                print(f"Updated {coin}: added {len(df_new)} new candles")

# Example usage
if __name__ == '__main__':
    collector = DataCollector()
    
    # Initial data collection
    data = collector.collect_all_symbols(start_date='2025-01-01')
    
    print("\nData collection complete!")
    for coin, df in data.items():
        print(f"{coin}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")