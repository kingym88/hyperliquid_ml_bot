import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import signal
import sys
import pandas as pd

from data_collector import DataCollector
from ml_models import TradingMLModel
from hyperliquid_trader import HyperliquidTrader
from performance_tracker import PerformanceTracker
from self_learning import SelfLearningSystem

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.data_collector = DataCollector()
        self.performance_tracker = PerformanceTracker()
        
        # Initialize trader
        address = os.getenv('HYPERLIQUID_ADDRESS')
        private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        use_testnet = os.getenv('USE_TESTNET', 'true').lower() == 'true'
        
        self.trader = HyperliquidTrader(address, private_key, use_testnet)
        
        # Trading configuration
        self.position_size_usd = float(os.getenv('POSITION_SIZE_USD', 100))
        self.max_positions = int(os.getenv('MAX_POSITIONS', 3))
        self.min_confidence = 0.50  # 60% minimum confidence
        self.lookback_minutes = int(os.getenv('LOOKBACK_PERIOD_MINUTES', 60))
        
        # Load ML models
        self.models = {}
        self.coins = ['HBAR', 'XRP', 'XLM']
        self.load_models()
        
        # Initialize self-learning
        self.learning_system = SelfLearningSystem(
            self.data_collector, 
            self.performance_tracker
        )
        
        # Track open trades
        self.open_trades = {}  # {coin: trade_id}
        
        self.running = False
    
    def load_models(self):
        """Load trained ML models"""
        print("\nLoading ML models...")
        
        for coin in self.coins:
            try:
                model = TradingMLModel(coin, self.lookback_minutes)
                model.load()
                self.models[coin] = model
                print(f"  ✓ {coin} model loaded")
            except FileNotFoundError:
                print(f"  ✗ {coin} model not found. Please train first.")
                sys.exit(1)
    
    def get_recent_data(self, coin, minutes=120):
        """Get recent price data for analysis"""
        df = self.data_collector.load_saved_data(coin)
        
        if df.empty:
            return None
        
        # Get last N minutes
        cutoff = datetime.now() - timedelta(minutes=minutes)
        df = df[df.index >= cutoff]
        
        return df
    
    def generate_signals(self):
        """Generate trading signals for all coins"""
        signals = {}
        
        # Get recent BTC data
        btc_data = self.get_recent_data('BTC', minutes=120)
        
        if btc_data is None or btc_data.empty:
            print("Warning: No recent BTC data available")
            return signals
        
        # Generate signal for each coin
        for coin in self.coins:
            try:
                alt_data = self.get_recent_data(coin, minutes=120)
                
                if alt_data is None or alt_data.empty:
                    print(f"Warning: No recent data for {coin}")
                    continue
                
                # Get prediction
                prediction = self.models[coin].predict(btc_data, alt_data)
                
                if prediction:
                    signals[coin] = prediction
                    
                    # Log prediction for later evaluation
                    self.performance_tracker.log_prediction(
                        coin=coin,
                        signal=prediction['signal'],
                        confidence=prediction['confidence']
                    )
                
            except Exception as e:
                print(f"Error generating signal for {coin}: {e}")
        
        return signals
    
    def execute_signals(self, signals):
        """Execute trading signals"""
        # Count current positions
        current_positions = len(self.trader.get_positions())
        
        # Sort signals by confidence (highest first)
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        for coin, signal in sorted_signals:
            # Check if we've reached max positions
            if current_positions >= self.max_positions:
                print(f"Max positions ({self.max_positions}) reached. Skipping {coin}.")
                break
            
            # Execute signal
            result = self.trader.execute_signal(
                coin=coin,
                signal=signal['signal'],
                confidence=signal['confidence'],
                min_confidence=self.min_confidence,
                position_size_usd=self.position_size_usd
            )
            
            # Log trade if executed
            if result.get('status') == 'success':
                entry_price = result.get('price')
                size = result.get('size')
                
                trade_id = self.performance_tracker.log_trade_entry(
                    coin=coin,
                    signal=signal['signal'],
                    confidence=signal['confidence'],
                    entry_price=entry_price,
                    position_size=size
                )
                
                self.open_trades[coin] = trade_id
                current_positions += 1
    
    def check_exit_conditions(self):
        """Check if we should exit any positions"""
        positions = self.trader.get_positions()
        
        for position in positions:
            coin = position['coin']
            
            # Skip if not in our trading universe
            if coin not in self.coins:
                continue
            
            # Get current price
            current_price = self.trader.get_market_price(coin)
            if not current_price:
                continue
            
            entry_price = position['entry_price']
            pnl_percent = (position['unrealized_pnl'] / 
                          (entry_price * abs(position['size']))) * 100
            
            # Exit conditions
            should_exit = False
            exit_reason = ''
            
            # Stop loss: -2%
            if pnl_percent <= -2.0:
                should_exit = True
                exit_reason = 'stop_loss'
            
            # Take profit: +3%
            elif pnl_percent >= 3.0:
                should_exit = True
                exit_reason = 'take_profit'
            
            # Time-based exit: Hold max 4 hours
            elif coin in self.open_trades:
                trade_data = self.performance_tracker.get_open_trades(coin)
                if not trade_data.empty:
                    entry_time = pd.to_datetime(trade_data.iloc[0]['timestamp'])
                    hours_held = (datetime.now() - entry_time).total_seconds() / 3600
                    
                    if hours_held >= 4:
                        should_exit = True
                        exit_reason = 'time_limit'
            
            if should_exit:
                print(f"\nExiting {coin} position ({exit_reason}): PnL {pnl_percent:+.2f}%")
                
                # Close position
                self.trader.close_position(coin)
                
                # Log exit
                if coin in self.open_trades:
                    self.performance_tracker.log_trade_exit(
                        self.open_trades[coin],
                        current_price
                    )
                    del self.open_trades[coin]
    
    def trading_loop(self):
        """Main trading loop"""
        print("\n" + "="*70)
        print("TRADING BOT STARTED")
        print("="*70)
        print(f"Coins: {', '.join(self.coins)}")
        print(f"Position Size: ${self.position_size_usd}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Min Confidence: {self.min_confidence:.0%}")
        print(f"Loop Interval: 1 minute")
        print("="*70 + "\n")
        
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                print(f"\n{'─'*70}")
                print(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'─'*70}")
                
                # Update data with latest candles
                if iteration % 5 == 0:  # Every 5 minutes
                    print("Updating price data...")
                    self.data_collector.update_data()
                
                # Check exit conditions for open positions
                self.check_exit_conditions()
                
                # Generate new signals
                print("\nGenerating signals...")
                signals = self.generate_signals()
                
                if signals:
                    print("\nSignals:")
                    for coin, signal in signals.items():
                        print(f"  {coin}: {signal['signal']} "
                              f"(Confidence: {signal['confidence']:.2%})")
                    
                    # Execute signals
                    self.execute_signals(signals)
                else:
                    print("No signals generated.")
                
                # Show current status
                account_value = self.trader.get_account_value()
                positions = self.trader.get_positions()
                
                print(f"\nAccount Value: ${account_value:,.2f}")
                print(f"Open Positions: {len(positions)}")
                
                for pos in positions:
                    pnl_pct = (pos['unrealized_pnl'] / 
                             (pos['entry_price'] * abs(pos['size']))) * 100
                    print(f"  {pos['coin']}: {pos['size']:+.4f} "
                          f"@ ${pos['entry_price']:.4f} "
                          f"(PnL: {pnl_pct:+.2f}%)")
                
                # Performance summary every 10 iterations
                if iteration % 10 == 0:
                    self.performance_tracker.generate_report()
                
                # Wait 1 minute
                print(f"\nWaiting 60 seconds...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n\nShutdown requested...")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        
        # Start self-learning system
        self.learning_system.start()
        
        # Start trading loop
        try:
            self.trading_loop()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        print("\nStopping bot...")
        self.running = False
        
        # Stop self-learning
        self.learning_system.stop()
        
        # Close all positions (optional)
        # Uncomment if you want to close all positions on shutdown
        # for coin in self.coins:
        #     self.trader.close_position(coin)
        
        # Final performance report
        print("\nFinal Performance Report:")
        self.performance_tracker.generate_report()
        
        # Close database
        self.performance_tracker.close()
        
        print("\nBot stopped.")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nShutdown signal received...')
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start bot
    bot = TradingBot()
    bot.start()