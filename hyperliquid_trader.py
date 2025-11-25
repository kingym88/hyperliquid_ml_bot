from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import json
from eth_account import Account
import time

class HyperliquidTrader:
    """Manages trading on Hyperliquid exchange"""

    def __init__(self, address, private_key, use_testnet=True):
        """
        Initialize Hyperliquid trader
        Args:
            address: Your main wallet address (0x...)
            private_key: Your API wallet private key (0x...)
            use_testnet: Use testnet (True) or mainnet (False)
        """
        self.address = address
        self.account = Account.from_key(private_key)
        base_url = (constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL)
        self.info = Info(base_url, skip_ws=True)
        self.exchange = Exchange(self.account, base_url)
        print(f"Connected to Hyperliquid ({'TESTNET' if use_testnet else 'MAINNET'})")
        self.symbol_map = {'HBAR': 'HBAR', 'XRP': 'XRP', 'XLM': 'XLM', 'BTC': 'BTC'}
        # Tick/size precision for supported coins: update if Hyperliquid changes these!
        self.size_precision = {
            'HBAR': {'decimals': 0, 'step': 1, 'tick': 0.001, 'min_size': 1},
            'XRP': {'decimals': 0, 'step': 1, 'tick': 0.001, 'min_size': 1},
            'XLM': {'decimals': 0, 'step': 10, 'tick': 0.0001, 'min_size': 10}
        }

    def quantize_size(self, coin, size):
        info = self.size_precision.get(coin, {'step': 1, 'decimals': 0, 'min_size': 1})
        step = info['step']
        decimals = info['decimals']
        min_size = info['min_size']
        quantized = int(size / step) * step  # always round DOWN
        quantized = round(quantized, decimals)
        if quantized < min_size:
            quantized = 0  # do not execute if below minimum
        return quantized

    def quantize_price(self, coin, price):
        tick = self.size_precision.get(coin, {}).get('tick', 0.001)
        # price must be a multiple of tick size, rounded to 6 decimals for safety
        return round(round(price / tick) * tick, 6)

    def get_account_state(self):
        """Get current account state"""
        try:
            state = self.info.user_state(self.address)
            return state
        except Exception as e:
            print(f"Error getting account state: {e}")
            return None

    def get_positions(self):
        """Get current open positions"""
        try:
            state = self.get_account_state()
            if not state:
                return []
            positions = []
            for position in state.get('assetPositions', []):
                pos_data = position['position']
                if float(pos_data['szi']) != 0: # Non-zero position
                    positions.append({
                        'coin': pos_data['coin'],
                        'size': float(pos_data['szi']),
                        'entry_price': float(pos_data['entryPx']),
                        'unrealized_pnl': float(pos_data['unrealizedPnl']),
                        'leverage': float(pos_data.get('leverage', {}).get('value', 1))
                    })
            return positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_account_value(self):
        """Get total account value in USD"""
        try:
            state = self.get_account_state()
            if not state:
                return 0
            margin_summary = state.get('marginSummary', {})
            account_value = float(margin_summary.get('accountValue', 0))
            return account_value
        except Exception as e:
            print(f"Error getting account value: {e}")
            return 0

    def get_market_price(self, coin):
        """Get current market price for a coin"""
        try:
            all_mids = self.info.all_mids()
            symbol = self.symbol_map.get(coin, coin)
            if symbol in all_mids:
                return float(all_mids[symbol])
            else:
                print(f"Price not found for {symbol}")
                return None
        except Exception as e:
            print(f"Error getting market price for {coin}: {e}")
            return None

    def calculate_position_size(self, coin, account_value, risk_per_trade=0.02, max_position_usd=None):
        """
        Calculate position size based on account value and risk
        """
        price = self.get_market_price(coin)
        if not price:
            return 0
        position_value = account_value * risk_per_trade
        if max_position_usd and position_value > max_position_usd:
            position_value = max_position_usd
        size = position_value / price
        size = self.quantize_size(coin, size)  # ensures valid lot
        return size

    def place_order(self, coin, is_buy, size, order_type='market', limit_price=None, reduce_only=False):
        """
        Place an order on Hyperliquid.
        Ensures size and price are correctly quantized for the coin.
        """
        try:
            symbol = self.symbol_map.get(coin, coin)
            if order_type == 'limit' and limit_price is None:
                limit_price = self.get_market_price(coin)
                if not limit_price:
                    return {'status': 'error', 'error': 'Could not get market price'}

            if order_type == 'market':
                current_price = self.get_market_price(coin)
                if not current_price:
                    return {'status': 'error', 'error': 'Could not get market price'}
                slippage = 0.01
                limit_price = (current_price * (1 + slippage) if is_buy else current_price * (1 - slippage))

            # Quantize price and size
            limit_price = self.quantize_price(coin, limit_price)
            size = self.quantize_size(coin, size)
            min_size = self.size_precision[coin]['min_size']

            if size < min_size:
                print(f"Order size {size} below minimum {min_size}, skipping order.")
                return {'status': 'error', 'error': 'size below minimum'}

            order_result = self.exchange.order(
                symbol,              # str: Coin ticker
                is_buy,              # bool: True for buy/long
                size,                # float: position size
                limit_price,         # float: price
                {"limit": {"tif": "Ioc"}},  # dict: order type
                reduce_only          # bool: reduce only
            )
            print(f"\nOrder placed: {'BUY' if is_buy else 'SELL'} {size} {coin} @ ${limit_price}")
            print(f"Result: {order_result}")

            return {
                'status': 'success',
                'coin': coin,
                'side': 'BUY' if is_buy else 'SELL',
                'size': size,
                'price': limit_price,
                'result': order_result
            }

        except Exception as e:
            print(f"Error placing order: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def close_position(self, coin):
        """Close an existing position"""
        try:
            positions = self.get_positions()
            position = None
            for pos in positions:
                if pos['coin'] == coin:
                    position = pos
                    break
            if not position:
                print(f"No open position for {coin}")
                return {'status': 'no_position'}
            is_buy = position['size'] < 0  # Short → buy to close
            size = abs(position['size'])
            result = self.place_order(
                coin=coin,
                is_buy=is_buy,
                size=size,
                order_type='market',
                reduce_only=True
            )
            return result
        except Exception as e:
            print(f"Error closing position: {e}")
            return {'status': 'error', 'error': str(e)}

    def execute_signal(self, coin, signal, confidence, min_confidence=0.6, position_size_usd=100):
        """Execute a trading signal"""
        print(f"\n{'='*60}")
        print(f"Signal: {signal} {coin} (Confidence: {confidence:.2%})")
        print(f"{'='*60}")

        if confidence < min_confidence:
            print(f"Confidence {confidence:.2%} below minimum {min_confidence:.2%}")
            return {'status': 'skipped', 'reason': 'low_confidence'}
        positions = self.get_positions()
        current_position = None
        for pos in positions:
            if pos['coin'] == coin:
                current_position = pos
                break
        if current_position:
            current_direction = 'LONG' if current_position['size'] > 0 else 'SHORT'
            if current_direction == signal:
                print(f"Already in {signal} position for {coin}")
                return {'status': 'already_positioned'}
            else:
                print(f"Closing existing {current_direction} position...")
                close_result = self.close_position(coin)
                time.sleep(2)
        account_value = self.get_account_value()
        size = self.calculate_position_size(
            coin=coin,
            account_value=account_value,
            risk_per_trade=0.02,
            max_position_usd=position_size_usd
        )
        if size == 0:
            print(f"Calculated size {size} is below minimum, not sending order.")
            return {'status': 'error', 'reason': 'invalid_size'}
        is_buy = (signal == 'LONG')
        result = self.place_order(
            coin=coin,
            is_buy=is_buy,
            size=size,
            order_type='market'
        )
        return result

# Example usage
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()
    address = os.getenv('HYPERLIQUID_ADDRESS')
    private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
    use_testnet = os.getenv('USE_TESTNET', 'true').lower() == 'true'
    trader = HyperliquidTrader(address, private_key, use_testnet)
    account_value = trader.get_account_value()
    print(f"\nAccount Value: ${account_value:,.2f}")
    positions = trader.get_positions()
    print(f"\nOpen Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos['coin']}: {pos['size']:+.4f} @ ${pos['entry_price']:.4f} (PnL: ${pos['unrealized_pnl']:+.2f})")
    for coin in ['HBAR', 'XRP', 'XLM']:
        price = trader.get_market_price(coin)
        if price is not None:
            print(f"\n{coin} Price: ${price:.4f}")
        else:
            print(f"\n{coin} Price: [unavailable - check symbol and/or network]")
