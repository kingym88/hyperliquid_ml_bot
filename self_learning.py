from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import os

class SelfLearningSystem:
    """Manages model retraining and improvement"""
    
    def __init__(self, data_collector, performance_tracker):
        self.data_collector = data_collector
        self.performance_tracker = performance_tracker
        self.scheduler = BackgroundScheduler()
        self.retraining_interval_days = 14
    
    def start(self):
        """Start the self-learning scheduler"""
        # Schedule retraining every 14 days
        self.scheduler.add_job(
            self.retrain_all_models,
            'interval',
            days=self.retraining_interval_days,
            next_run_time=datetime.now() + timedelta(days=self.retraining_interval_days)
        )
        
        self.scheduler.start()
        print(f"Self-learning system started. Next retraining in {self.retraining_interval_days} days.")
    
    def retrain_all_models(self):
        """Retrain all models with latest data"""
        from ml_models import TradingMLModel
        
        print("\n" + "="*70)
        print("STARTING SELF-LEARNING RETRAINING")
        print(f"Timestamp: {datetime.now()}")
        print("="*70)
        
        # Update data with latest prices
        print("\nStep 1: Updating price data...")
        self.data_collector.update_data()
        
        # Load BTC data
        btc_df = self.data_collector.load_saved_data('BTC')
        
        if btc_df.empty:
            print("Error: No BTC data available")
            return
        
        coins = ['HBAR', 'XRP', 'XLM']
        
        for coin in coins:
            print(f"\n{'='*70}")
            print(f"Retraining {coin} Model")
            print(f"{'='*70}")
            
            # Get performance summary for this coin
            perf = self.performance_tracker.get_performance_summary(
                coin=coin, 
                days=self.retraining_interval_days
            )
            
            print(f"\nRecent Performance ({self.retraining_interval_days} days):")
            print(f"  Trades: {perf['total_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.2%}")
            print(f"  Total PnL: ${perf['total_pnl']:+,.2f}")
            
            # Load altcoin data
            alt_df = self.data_collector.load_saved_data(coin)
            
            if alt_df.empty:
                print(f"Error: No data available for {coin}")
                continue
            
            # Initialize new model
            model = TradingMLModel(coin)
            
            # Train with updated data
            metrics = model.train(btc_df, alt_df, test_size=0.2)
            
            # Save updated model
            model.save()
            
            # Log retraining
            self.performance_tracker.log_retraining(
                coin=coin,
                samples_used=metrics['n_samples'],
                accuracy=metrics['accuracy'],
                notes=f"14-day retrain. Recent win rate: {perf['win_rate']:.2%}"
            )
            
            print(f"\n{coin} model retrained successfully!")
        
        print("\n" + "="*70)
        print("RETRAINING COMPLETE")
        print("="*70 + "\n")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        print("Self-learning system stopped.")

# Example usage
if __name__ == '__main__':
    from data_collector import DataCollector
    from performance_tracker import PerformanceTracker
    
    collector = DataCollector()
    tracker = PerformanceTracker()
    
    learning = SelfLearningSystem(collector, tracker)
    learning.start()
    
    print("System running. Press Ctrl+C to stop.")
    
    try:
        while True:
            import time
            time.sleep(60)
    except KeyboardInterrupt:
        learning.stop()
        print("Stopped.")