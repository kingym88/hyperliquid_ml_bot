#!/bin/bash

# Hyperliquid ML Trading Bot - Monitoring Script

echo "=============================================="
echo "   Hyperliquid ML Trading Bot Monitor"
echo "=============================================="
echo ""

# Check if bot is running
echo "=== Bot Status ==="
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✓ Bot is RUNNING"
    PID=$(pgrep -f "python.*main.py")
    echo "  Process ID: $PID"
    
    # Get runtime
    START_TIME=$(ps -p $PID -o lstart=)
    echo "  Started: $START_TIME"
else
    echo "✗ Bot is STOPPED"
fi
echo ""

# Check virtual environment
echo "=== Environment ==="
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
else
    echo "✗ Virtual environment not found"
fi
echo ""

# Check data files
echo "=== Data Files ==="
for coin in BTC HBAR XRP XLM; do
    FILE="data/${coin}_1m.csv"
    if [ -f "$FILE" ]; then
        SIZE=$(ls -lh "$FILE" | awk '{print $5}')
        LINES=$(wc -l < "$FILE")
        echo "✓ $coin: $SIZE ($LINES rows)"
    else
        echo "✗ $coin: Missing"
    fi
done
echo ""

# Check model files
echo "=== Model Files ==="
for coin in HBAR XRP XLM; do
    MODEL="models/${coin}_model.pkl"
    if [ -f "$MODEL" ]; then
        echo "✓ $coin: Model trained"
    else
        echo "✗ $coin: Model missing"
    fi
done
echo ""

# Check performance database
echo "=== Performance Database ==="
if [ -f "data/performance.db" ]; then
    echo "✓ Performance database exists"
    
    # Query trade count if sqlite3 available
    if command -v sqlite3 &> /dev/null; then
        TOTAL_TRADES=$(sqlite3 data/performance.db "SELECT COUNT(*) FROM trades;" 2>/dev/null)
        OPEN_TRADES=$(sqlite3 data/performance.db "SELECT COUNT(*) FROM trades WHERE status='open';" 2>/dev/null)
        CLOSED_TRADES=$(sqlite3 data/performance.db "SELECT COUNT(*) FROM trades WHERE status='closed';" 2>/dev/null)
        
        echo "  Total trades: $TOTAL_TRADES"
        echo "  Open: $OPEN_TRADES"
        echo "  Closed: $CLOSED_TRADES"
    fi
else
    echo "○ Performance database not yet created (bot hasn't run)"
fi
echo ""

# Show recent performance
echo "=== Recent Performance ==="
if [ -f "data/performance.db" ] && command -v sqlite3 &> /dev/null; then
    python3 << 'EOF'
from performance_tracker import PerformanceTracker
import sys

try:
    tracker = PerformanceTracker()
    
    # Quick summary
    summary = tracker.get_performance_summary()
    
    if summary['total_trades'] > 0:
        print(f"  Total Trades: {summary['total_trades']}")
        print(f"  Win Rate: {summary['win_rate']:.1%}")
        print(f"  Total PnL: ${summary['total_pnl']:+,.2f}")
        print(f"  Avg PnL: ${summary['avg_pnl']:+.2f}")
    else:
        print("  No trades yet")
    
    tracker.close()
except Exception as e:
    print(f"  Could not load performance data: {e}")
EOF
else
    echo "  Performance tracking not available"
fi
echo ""

# Show disk usage
echo "=== Disk Usage ==="
du -sh data/ 2>/dev/null || echo "  Data directory not found"
du -sh models/ 2>/dev/null || echo "  Models directory not found"
echo ""

# Show recent logs (if log file exists)
echo "=== Recent Activity ==="
if [ -f "logs/trading.log" ]; then
    echo "Last 5 log entries:"
    tail -5 logs/trading.log
else
    echo "No log file found (logs/trading.log)"
fi
echo ""

# System resources (if bot is running)
if pgrep -f "python.*main.py" > /dev/null; then
    echo "=== System Resources ==="
    PID=$(pgrep -f "python.*main.py")
    
    # CPU and Memory
    PS_INFO=$(ps -p $PID -o %cpu,%mem,vsz,rss | tail -1)
    echo "  $PS_INFO"
    echo "  (CPU% MEM% VSZ RSS)"
    echo ""
fi

# Network connectivity test
echo "=== Network Status ==="
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo "✓ Internet connection active"
else
    echo "✗ No internet connection"
fi
echo ""

# Quick tips
echo "=== Quick Commands ==="
echo "  Start bot:     python main.py"
echo "  Stop bot:      Press Ctrl+C in bot terminal"
echo "  Update data:   python data_collector.py"
echo "  Retrain:       python ml_models.py"
echo "  Full report:   python -c 'from performance_tracker import PerformanceTracker; t=PerformanceTracker(); t.generate_report(); t.close()'"
echo ""

echo "=============================================="
echo "              Monitor Complete"
echo "=============================================="