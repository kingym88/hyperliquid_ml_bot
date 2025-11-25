import sqlite3
import pandas as pd
from datetime import datetime
import os

class PerformanceTracker:
    """Track trading performance and calculate metrics"""
    
    def __init__(self, db_path='data/performance.db'):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        
        # Create trades table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                coin TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                position_size REAL NOT NULL,
                exit_price REAL,
                exit_timestamp TEXT,
                pnl REAL,
                pnl_percent REAL,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        # Create predictions table (for model performance)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                coin TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                actual_outcome TEXT,
                correct BOOLEAN
            )
        ''')
        
        # Create retraining log
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS retraining_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                coin TEXT NOT NULL,
                samples_used INTEGER,
                accuracy REAL,
                notes TEXT
            )
        ''')
        
        self.conn.commit()
    
    def log_trade_entry(self, coin, signal, confidence, entry_price, position_size):
        """Log a new trade entry"""
        timestamp = datetime.now().isoformat()
        
        self.conn.execute('''
            INSERT INTO trades 
            (timestamp, coin, signal, confidence, entry_price, position_size, status)
            VALUES (?, ?, ?, ?, ?, ?, 'open')
        ''', (timestamp, coin, signal, confidence, entry_price, position_size))
        
        self.conn.commit()
        
        print(f"Trade logged: {signal} {coin} @ ${entry_price:.4f}")
        
        return self.conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    
    def log_trade_exit(self, trade_id, exit_price):
        """Log a trade exit and calculate PnL"""
        exit_timestamp = datetime.now().isoformat()
        
        # Get trade details
        trade = self.conn.execute('''
            SELECT entry_price, position_size, signal 
            FROM trades WHERE id = ?
        ''', (trade_id,)).fetchone()
        
        if not trade:
            print(f"Trade {trade_id} not found")
            return
        
        entry_price, position_size, signal = trade
        
        # Calculate PnL
        if signal == 'LONG':
            pnl = (exit_price - entry_price) * position_size
        else:  # SHORT
            pnl = (entry_price - exit_price) * position_size
        
        pnl_percent = (pnl / (entry_price * position_size)) * 100
        
        # Update trade
        self.conn.execute('''
            UPDATE trades 
            SET exit_price = ?, exit_timestamp = ?, pnl = ?, 
                pnl_percent = ?, status = 'closed'
            WHERE id = ?
        ''', (exit_price, exit_timestamp, pnl, pnl_percent, trade_id))
        
        self.conn.commit()
        
        print(f"Trade {trade_id} closed: PnL ${pnl:+.2f} ({pnl_percent:+.2f}%)")
    
    def get_open_trades(self, coin=None):
        """Get all open trades"""
        query = "SELECT * FROM trades WHERE status = 'open'"
        params = []
        
        if coin:
            query += " AND coin = ?"
            params.append(coin)
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_performance_summary(self, coin=None, days=None):
        """Get performance summary"""
        query = "SELECT * FROM trades WHERE status = 'closed'"
        params = []
        
        if coin:
            query += " AND coin = ?"
            params.append(coin)
        
        if days:
            query += f" AND datetime(timestamp) > datetime('now', '-{days} days')"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_pnl_percent': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pnl_percent': df['pnl_percent'].mean(),
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min()
        }
    
    def log_prediction(self, coin, signal, confidence):
        """Log a prediction for later evaluation"""
        timestamp = datetime.now().isoformat()
        
        self.conn.execute('''
            INSERT INTO predictions 
            (timestamp, coin, signal, confidence)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, coin, signal, confidence))
        
        self.conn.commit()
    
    def log_retraining(self, coin, samples_used, accuracy, notes=''):
        """Log model retraining event"""
        timestamp = datetime.now().isoformat()
        
        self.conn.execute('''
            INSERT INTO retraining_log 
            (timestamp, coin, samples_used, accuracy, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, coin, samples_used, accuracy, notes))
        
        self.conn.commit()
        
        print(f"Retraining logged for {coin}: {accuracy:.2%} accuracy")
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*70)
        print("TRADING PERFORMANCE REPORT")
        print("="*70)
        
        for coin in ['HBAR', 'XRP', 'XLM']:
            print(f"\n{coin}:")
            print("-" * 50)
            
            summary = self.get_performance_summary(coin=coin)
            
            print(f"  Total Trades: {summary['total_trades']}")
            print(f"  Win Rate: {summary['win_rate']:.2%}")
            print(f"  Total PnL: ${summary['total_pnl']:+,.2f}")
            print(f"  Avg PnL: ${summary['avg_pnl']:+.2f} "
                  f"({summary['avg_pnl_percent']:+.2f}%)")
            
            if summary['total_trades'] > 0:
                print(f"  Best Trade: ${summary['best_trade']:+.2f}")
                print(f"  Worst Trade: ${summary['worst_trade']:+.2f}")
        
        # Overall summary
        overall = self.get_performance_summary()
        print(f"\n{'='*70}")
        print("OVERALL:")
        print(f"  Total Trades: {overall['total_trades']}")
        print(f"  Win Rate: {overall['win_rate']:.2%}")
        print(f"  Total PnL: ${overall['total_pnl']:+,.2f}")
        print(f"{'='*70}\n")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Example usage
if __name__ == '__main__':
    tracker = PerformanceTracker()
    
    # Simulate some trades
    trade_id = tracker.log_trade_entry('HBAR', 'LONG', 0.75, 0.25, 1000)
    tracker.log_trade_exit(trade_id, 0.26)
    
    # Generate report
    tracker.generate_report()
    
    tracker.close()