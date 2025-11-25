# Hyperliquid ML Trading Bot

An automated machine learning trading bot that predicts long/short positions for HBAR [finance:Hedera Hashgraph], XRP [finance:Ripple], and XLM [finance:Stellar] based on Bitcoin [finance:Bitcoin] price correlations. The bot trades on Hyperliquid perpetual futures and self-improves every 14 days.

## 🚀 Features

- **Real-time ML Predictions**: Minute-by-minute signal generation using Gradient Boosting
- **Independent Coin Models**: Separate models for HBAR-BTC, XRP-BTC, XLM-BTC correlations
- **Automated Trading**: Executes trades on Hyperliquid with configurable risk management
- **Performance Tracking**: SQLite database logs all trades and calculates metrics
- **Self-Learning**: Automatically retrains models every 14 days with recent data
- **Risk Controls**: Stop loss (-2%), take profit (+3%), time limits (4 hours), position sizing

## 📋 Prerequisites

- Mac with VS Code 2
- Python 3.11 or later
- Hyperliquid account (testnet or mainnet)
- Internet connection

## 🛠️ Quick Start

### 1. Setup

```bash
# Create project directory
mkdir hyperliquid_ml_bot && cd hyperliquid_ml_bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Create directory structure
mkdir -p data models logs config

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your Hyperliquid credentials
```

### 2. Collect Data

```bash
python data_collector.py
```

Downloads historical minute-by-minute price data for BTC, HBAR, XRP, XLM (takes 5-10 minutes).

### 3. Train Models

```bash
python ml_models.py
```

Trains 3 ML models using historical data (takes 5-10 minutes).

### 4. Test Connection

```bash
python hyperliquid_trader.py
```

Verifies Hyperliquid API connection and shows account status.

### 5. Run Bot

```bash
python main.py
```

Starts the trading bot. Monitor closely for first hour!

## 📁 File Structure

```
hyperliquid_ml_bot/
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .env                      # Your credentials (SECRET!)
├── data_collector.py         # Data collection module
├── ml_models.py              # ML training/prediction
├── hyperliquid_trader.py     # Trading execution
├── performance_tracker.py    # Performance tracking
├── self_learning.py          # Auto-retraining system
├── main.py                   # Main bot orchestrator
├── monitor.sh                # Monitoring script
├── SETUP_GUIDE.md           # Detailed setup instructions
├── data/                     # Price data & database
├── models/                   # Trained ML models
└── logs/                     # Trading logs
```

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Hyperliquid
HYPERLIQUID_ADDRESS=0xYour...      # Your wallet address
HYPERLIQUID_PRIVATE_KEY=0xYour...  # API wallet private key
USE_TESTNET=true                    # true for testnet, false for mainnet

# Trading
POSITION_SIZE_USD=100               # Max position size per trade
MAX_POSITIONS=3                     # Max concurrent positions
RISK_PER_TRADE=0.02                # 2% risk per trade

# Data
DATA_START_DATE=2025-01-01         # Historical data start
LOOKBACK_PERIOD_MINUTES=60         # Feature lookback window
```

## 🎯 How It Works

### Data Flow

1. **Data Collection**: Fetches minute-by-minute OHLCV data from Binance
2. **Feature Engineering**: Calculates 60+ technical indicators (RSI, MACD, correlations, etc.)
3. **ML Prediction**: Gradient Boosting models predict long/short signals
4. **Signal Filtering**: Only executes trades with x%+ confidence (set to 50% to begin)
5. **Trade Execution**: Places orders on Hyperliquid with risk management
6. **Performance Tracking**: Logs trades, calculates PnL, tracks metrics
7. **Self-Learning**: Retrains models every 14 days with latest data

### ML Features

Each model analyzes:
- BTC price changes (1m, 5m, 15m, 60m)
- BTC momentum indicators (RSI, MACD)
- Rolling correlation (60-minute window)
- Altcoin technical indicators
- Time-based features (hour, day of week)

### Risk Management

- **Position Sizing**: 2% of account per trade (configurable)
- **Stop Loss**: Automatic exit at -2%
- **Take Profit**: Automatic exit at +3%
- **Time Limit**: Maximum 4-hour hold time
- **Confidence Filter**: Minimum x% ML confidence (set to 50% to begin)
- **Max Positions**: Limits concurrent trades

## 📊 Monitoring

### Quick Status Check

```bash
./monitor.sh
```

Shows bot status, data files, model files, recent performance, and system resources.

### Performance Report

```bash
python -c "
from performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
tracker.generate_report()
tracker.close()
"
```

### View Recent Trades

```bash
sqlite3 data/performance.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"
```

## 🔧 Maintenance

### Update Data

```bash
python data_collector.py  # Full recollection
# OR
python -c "from data_collector import DataCollector; DataCollector().update_data()"  # Incremental
```

### Retrain Models

```bash
python ml_models.py
```

### Background Operation

Use `screen` for 24/7 operation:

```bash
# Start screen session
screen -S trading_bot

# Run bot
python main.py

# Detach: Press Ctrl+A, then D

# Reattach later
screen -r trading_bot
```

## 🛡️ Safety First

### ⚠️ Critical Warnings

1. **Start with Testnet**: Always test on Hyperliquid testnet first (24-48 hours minimum)
2. **Small Positions**: Begin with $10-20 positions, gradually increase
3. **Monitor Closely**: Watch the bot for first few hours, check daily initially
4. **Risk Only What You Can Lose**: Never trade with money you can't afford to lose
5. **API Wallet Security**: Use API wallets (no withdrawal permissions)

### Best Practices

- Run on testnet for 1 week before mainnet
- Start with position size = $10-20
- Monitor win rate and PnL daily (first week)
- Review performance after 14-day retraining
- Keep `.env` secure and never commit to git
- Have a kill switch ready (Ctrl+C)

## 📈 Performance Expectations

### Target Metrics

- **Win Rate**: 55-65% (profitable with risk management)
- **Average PnL**: Positive over 14-day period
- **Model Accuracy**: Above 55% for reliable predictions
- **Sharpe Ratio**: Aim for >1.0 over monthly periods

### Realistic Expectations

- Some losing trades are normal (40-45% of trades)
- Short-term losses happen, focus on long-term performance
- Market conditions affect results (volatility, correlations)
- Self-learning improves performance over time

## 🐛 Troubleshooting

### "Model not trained yet"
```bash
python ml_models.py
```

### "No data available"
```bash
python data_collector.py
```

### "Authentication failed"
Check `HYPERLIQUID_PRIVATE_KEY` in `.env`, ensure API wallet authorized.

### Low accuracy (<55%)
- Collect more data (60+ days)
- Check data quality (no gaps)
- Adjust features in `ml_models.py`
- Try different ML algorithms

## 📚 Resources

- [Hyperliquid Documentation](https://hyperliquid.gitbook.io)
- [Hyperliquid Testnet](https://app.hyperliquid-testnet.xyz)
- [CCXT Documentation](https://docs.ccxt.com)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)

## 📝 License

This project is for educational purposes. Use at your own risk. Trading cryptocurrencies carries significant financial risk.

## ⚡ Quick Commands

```bash
# Activate environment
source venv/bin/activate

# Collect data
python data_collector.py

# Train models
python ml_models.py

# Test connection
python hyperliquid_trader.py

# Run bot
python main.py

# Monitor status
./monitor.sh

# Performance report
python -c "from performance_tracker import PerformanceTracker; t=PerformanceTracker(); t.generate_report(); t.close()"
```

## 🤝 Support

For detailed setup instructions, see `SETUP_GUIDE.md`.

---

**Remember**: Past performance does not guarantee future results. Always start with testnet and small positions!
