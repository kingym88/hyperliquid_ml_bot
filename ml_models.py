import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

class TradingMLModel:
    """ML model for predicting long/short signals"""
    
    def __init__(self, coin_name, lookback_minutes=60):
        self.coin_name = coin_name
        self.lookback_minutes = lookback_minutes
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for a DataFrame"""
        df = df.copy()
        
        # Price changes
        for period in [1, 5, 15, 60]:
            df[f'return_{period}m'] = df['close'].pct_change(period)
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_15'] = df['close'].rolling(window=15).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], period=14)
        
        # MACD
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        
        # Volatility
        df['volatility_15'] = df['close'].rolling(window=15).std()
        df['volatility_60'] = df['close'].rolling(window=60).std()
        
        # Volume indicators
        df['volume_ma_15'] = df['volume'].rolling(window=15).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_15']
        
        return df
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def prepare_features(self, btc_df, alt_df):
        """
        Prepare feature matrix combining BTC and altcoin data
        
        Args:
            btc_df: Bitcoin DataFrame with OHLCV
            alt_df: Altcoin DataFrame with OHLCV
        
        Returns:
            DataFrame with features and target
        """
        # Calculate indicators
        btc_df = self.calculate_technical_indicators(btc_df)
        alt_df = self.calculate_technical_indicators(alt_df)
        
        # Align timestamps
        df = pd.DataFrame(index=alt_df.index)
        
        # BTC features (prefixed with btc_)
        for col in ['return_1m', 'return_5m', 'return_15m', 'return_60m',
                   'rsi', 'macd', 'macd_signal', 'volatility_15']:
            df[f'btc_{col}'] = btc_df[col]
        
        # Altcoin features (prefixed with alt_)
        for col in ['return_1m', 'return_5m', 'return_15m', 
                   'rsi', 'macd', 'volatility_15']:
            df[f'alt_{col}'] = alt_df[col]
        
        # Correlation features
        df['correlation_60m'] = (
            btc_df['close']
            .pct_change()
            .rolling(window=60)
            .corr(alt_df['close'].pct_change())
        )
        
        # Price relationship
        df['price_ratio'] = alt_df['close'] / btc_df['close']
        df['price_ratio_change'] = df['price_ratio'].pct_change(fill_method=None)
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['minute'] = df.index.minute
        
        # Create target: future return
        # 1 = Long (price will go up), 0 = Short (price will go down)
        future_return = alt_df['close'].pct_change(5).shift(-5)
        df['target'] = (future_return > 0).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train(self, btc_df, alt_df, test_size=0.2):
        """
        Train the ML model
        
        Args:
            btc_df: Bitcoin price data
            alt_df: Altcoin price data
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*50}")
        print(f"Training model for {self.coin_name}")
        print(f"{'='*50}")
        
        # Prepare features
        df = self.prepare_features(btc_df, alt_df)
        
        # Separate features and target
        X = df.drop(columns=['target'])
        y = df['target']
        
        self.feature_columns = X.columns.tolist()
        
        # Time-based split (important for time series)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (Gradient Boosting)
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        
        print("\nTraining model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Short', 'Long']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'n_samples': len(X_train)
        }
    
    def predict(self, btc_current, alt_current):
        """
        Make prediction for current market conditions
        
        Args:
            btc_current: Recent BTC DataFrame (last 60+ minutes)
            alt_current: Recent altcoin DataFrame (last 60+ minutes)
        
        Returns:
            Dictionary with prediction and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        df = self.prepare_features(btc_current, alt_current)
        
        if df.empty:
            return None
        
        # Get latest row
        X = df[self.feature_columns].iloc[-1:]  # DataFrame slice (retains feature names)
        X_scaled = self.scaler.transform(X)     # Scaler preserves column alignment
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = probabilities[prediction]
        
        signal = 'LONG' if prediction == 1 else 'SHORT'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'timestamp': df.index[-1],
            'probabilities': {
                'short': probabilities[0],
                'long': probabilities[1]
            }
        }
    
    def save(self, model_dir='models'):
        """Save trained model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'{self.coin_name}_model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.coin_name}_scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_columns, 
                   os.path.join(model_dir, f'{self.coin_name}_features.pkl'))
        
        print(f"Model saved to {model_path}")
    
    def load(self, model_dir='models'):
        """Load trained model and scaler"""
        model_path = os.path.join(model_dir, f'{self.coin_name}_model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.coin_name}_scaler.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(
            os.path.join(model_dir, f'{self.coin_name}_features.pkl')
        )
        
        print(f"Model loaded from {model_path}")

# Training script
if __name__ == '__main__':
    from data_collector import DataCollector
    
    collector = DataCollector()
    
    # Load data
    btc_df = collector.load_saved_data('BTC')
    
    coins = ['HBAR', 'XRP', 'XLM']
    
    for coin in coins:
        print(f"\n\nProcessing {coin}...")
        alt_df = collector.load_saved_data(coin)
        
        if btc_df.empty or alt_df.empty:
            print(f"Data not available for {coin}")
            continue
        
        # Initialize and train model
        model = TradingMLModel(coin)
        metrics = model.train(btc_df, alt_df)
        
        # Save model
        model.save()
        
        print(f"\n{coin} model training complete!")