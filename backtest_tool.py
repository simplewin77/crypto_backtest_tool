import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

# 配置日志
LOG_FILE = "backtest_log.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'),  # 输出到文件，覆盖模式
                        logging.StreamHandler()  # 同时输出到控制台
                    ])

# --- 用户配置参数 (全部在这里配置) ---
# 基础配置
SYMBOL = "WIFUSDT"
DATA_FILE_PATH = "data/wifusdt_data.csv"  # 请修改为您的数据文件路径
INITIAL_BALANCE = 10000.0
START_DATE = "2025-05-07 21:47:18"  # 调整为与示例数据匹配的日期
END_DATE = "2025-05-27 20:47:18"    # 调整为与示例数据匹配的日期
TRADING_FEE_RATE = 0.0002  # 0.02%
SLIPPAGE_RATE = 0.0001     # 0.01%
TARGET_FREQ = '1H'         # 目标K线周期，可选 '1min', '5min', '1H' 等

# 开仓条件配置
ENTRY_CONDITIONS_COMBINATION = "AND"  # "AND" 或 "OR"
ENABLE_OI_TOTAL_INCREASE_ENTRY = True
OI_TOTAL_INCREASE_THRESHOLD_ENTRY = 0.03  # OI增加阈值 (例如 3%)
ENABLE_OI_LONG_INCREASE_ENTRY = True
OI_LONG_INCREASE_THRESHOLD_ENTRY = 0.05   # 多单OI增加阈值 (例如 5%)
ENABLE_OI_SHORT_DECREASE_ENTRY = False
OI_SHORT_DECREASE_THRESHOLD_ENTRY = 0.03  # 空单OI减少阈值 (例如 3%)
ENABLE_VOLUME_INCREASE_ENTRY = True
VOLUME_INCREASE_MULTIPLIER_ENTRY = 1.5    # 交易量增加倍数 (例如 1.5倍)
ENABLE_BUY_SELL_VOLUME_RATIO_ENTRY = True
BUY_SELL_VOLUME_RATIO_THRESHOLD_ENTRY = 1.2  # 买卖单比率阈值 (多头买入意愿更强)
OI_VOLUME_WINDOW_SIZE_ENTRY = 5             # 计算窗口大小
ENABLE_PRICE_COOPERATION_ENTRY = True
PRICE_COOPERATION_LOOKBACK_ENTRY = 10       # 价格配合回溯期

# 平仓条件配置
EXIT_CONDITIONS_COMBINATION = "OR"  # "AND" 或 "OR"
ENABLE_OI_TOTAL_DECREASE_EXIT = True
OI_TOTAL_DECREASE_THRESHOLD_EXIT = 0.02    # OI减少阈值 (例如 2%)
ENABLE_OI_LONG_DECREASE_EXIT = True
OI_LONG_DECREASE_THRESHOLD_EXIT = 0.03     # 多单OI减少阈值 (例如 3%)
ENABLE_OI_SHORT_INCREASE_EXIT = False
OI_SHORT_INCREASE_THRESHOLD_EXIT = 0.02    # 空单OI增加阈值 (例如 2%)
ENABLE_VOLUME_DECREASE_EXIT = True
VOLUME_DECREASE_MULTIPLIER_EXIT = 0.8      # 交易量减少倍数 (例如 0.8倍)
ENABLE_BUY_SELL_VOLUME_RATIO_EXIT = True
BUY_SELL_VOLUME_RATIO_THRESHOLD_EXIT = 0.8 # 买卖单比率阈值 (空头买入意愿减弱)
OI_VOLUME_WINDOW_SIZE_EXIT = 5             # 计算窗口大小
ENABLE_STOP_LOSS = True
STOP_LOSS_PERCENT = 0.03                   # 3% 止损
ENABLE_TAKE_PROFIT = True
TAKE_PROFIT_PERCENT = 0.10                 # 10% 止盈

# 风险管理
POSITION_SIZE_TYPE = "FIXED_PERCENT"  # "FIXED_AMOUNT", "FIXED_PERCENT"
FIXED_AMOUNT = 1000.0                 # 如果是固定金额，每次开仓金额
FIXED_PERCENT = 0.95                  # 如果是固定比例，每次开仓使用余额的百分比 (例如 0.95 代表95%)

# --- 模拟交易器类 (PaperTrader) ---
class PaperTrader:
    def __init__(self, initial_balance: float, fee_rate: float, slippage_rate: float, position_size_type: str, fixed_amount: float, fixed_percent: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.position = 0  # 0: 空仓, 1: 多仓
        self.entry_price = 0.0
        self.position_size = 0.0
        self.trade_logs = []
        self.balance_history = []
        self.timestamps = []
        self.position_size_type = position_size_type
        self.fixed_amount = fixed_amount
        self.fixed_percent = fixed_percent
        
    def open_long(self, price: float, timestamp):
        """开多仓"""
        if self.position == 0:
            actual_price = price * (1 + self.slippage_rate)

            # 根据风险管理设置计算仓位大小
            if self.position_size_type == "FIXED_AMOUNT":
                self.position_size = self.fixed_amount / actual_price
                if self.position_size * actual_price > self.balance:
                    logging.info(f"[{timestamp}] 余额不足，无法以固定金额开仓。当前余额: {self.balance:.2f}")
                    return False
            elif self.position_size_type == "FIXED_PERCENT":
                self.position_size = (self.balance * self.fixed_percent) / actual_price
            else:
                self.position_size = self.balance / actual_price

            if self.position_size <= 0:
                logging.info(f"[{timestamp}] 计算仓位为0或负数，无法开仓。")
                return False

            self.entry_price = actual_price
            self.position = 1
            
            fee_cost = self.position_size * actual_price * self.fee_rate
            self.balance -= fee_cost

            self.trade_logs.append({
                'timestamp': timestamp,
                'action': 'OPEN_LONG',
                'price': actual_price,
                'position_size': self.position_size,
                'balance_before': self.balance + fee_cost,
                'fee': fee_cost,
                'slippage': price * self.slippage_rate * self.position_size
            })
            
            logging.info(f"[{timestamp}] 开多仓: 价格={actual_price:.4f}, 数量={self.position_size:.4f}, 手续费={fee_cost:.4f}, 当前余额={self.balance:.2f}")
            return True
        return False
            
    def close_long(self, price: float, timestamp, reason: str = "SIGNAL"):
        """平多仓"""
        if self.position == 1:
            actual_price = price * (1 - self.slippage_rate)
            gross_pnl = (actual_price - self.entry_price) * self.position_size
            fee_cost = actual_price * self.position_size * self.fee_rate
            net_pnl = gross_pnl - fee_cost
            
            self.balance += (self.position_size * actual_price) - (self.position_size * self.entry_price) - fee_cost

            self.trade_logs.append({
                'timestamp': timestamp,
                'action': f'CLOSE_LONG_{reason}',
                'price': actual_price,
                'position_size': self.position_size,
                'entry_price': self.entry_price,
                'gross_pnl': gross_pnl,
                'fee': fee_cost,
                'net_pnl': net_pnl,
                'balance_after': self.balance
            })
            
            logging.info(f"[{timestamp}] 平多仓({reason}): 价格={actual_price:.4f}, 盈亏={net_pnl:.2f}, 手续费={fee_cost:.4f}, 当前余额={self.balance:.2f}")
            
            self.position = 0
            self.entry_price = 0.0
            self.position_size = 0.0
            
    def update_balance_history(self, current_price: float, timestamp):
        """更新资金曲线"""
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            current_balance = self.balance + unrealized_pnl
        else:
            current_balance = self.balance
            
        self.balance_history.append(current_balance)
        self.timestamps.append(timestamp)

# --- 策略信号判断函数 ---
def precompute_indicators(df: pd.DataFrame, entry_window: int, exit_window: int, price_lookback: int) -> pd.DataFrame:
    """预计算所有需要的滑动窗口指标"""
    logging.info("开始预计算指标...")

    if 'oi' in df.columns:
        df[f'oi_total_change_pct_{entry_window}'] = df['oi'].pct_change(periods=entry_window)
        df[f'oi_total_change_pct_{exit_window}'] = df['oi'].pct_change(periods=exit_window)

        if 'sum_taker_long_short_vol_ratio' in df.columns:
            df['long_weight'] = df['sum_taker_long_short_vol_ratio'].apply(
                lambda x: min(0.9, max(0.1, x / (x + 1))) if x is not None and x > 0 else 0.5
            )
            df['short_weight'] = 1 - df['long_weight']

            df[f'oi_long_change_pct_{entry_window}'] = df[f'oi_total_change_pct_{entry_window}'] * df['long_weight']
            df[f'oi_short_change_pct_{entry_window}'] = df[f'oi_total_change_pct_{entry_window}'] * df['short_weight']
            df[f'oi_long_change_pct_{exit_window}'] = df[f'oi_total_change_pct_{exit_window}'] * df['long_weight']
            df[f'oi_short_change_pct_{exit_window}'] = df[f'oi_total_change_pct_{exit_window}'] * df['short_weight']
        else:
            df[f'oi_long_change_pct_{entry_window}'] = df[f'oi_total_change_pct_{entry_window}'] * 0.5
            df[f'oi_short_change_pct_{entry_window}'] = df[f'oi_total_change_pct_{entry_window}'] * 0.5
            df[f'oi_long_change_pct_{exit_window}'] = df[f'oi_total_change_pct_{exit_window}'] * 0.5
            df[f'oi_short_change_pct_{exit_window}'] = df[f'oi_total_change_pct_{exit_window}'] * 0.5

    if 'volume' in df.columns:
        df[f'avg_volume_{entry_window}'] = df['volume'].rolling(window=entry_window, min_periods=1).mean()
        df[f'volume_ratio_{entry_window}'] = df['volume'] / df[f'avg_volume_{entry_window}']
        df[f'avg_volume_{exit_window}'] = df['volume'].rolling(window=exit_window, min_periods=1).mean()
        df[f'volume_ratio_{exit_window}'] = df['volume'] / df[f'avg_volume_{exit_window}']
        
        if 'taker_buy_volume' in df.columns:
            df['taker_sell_volume'] = df['volume'] - df['taker_buy_volume']
            df['buy_sell_ratio'] = df['taker_buy_volume'] / df['taker_sell_volume'].replace(0, np.nan)
            df['buy_sell_ratio'].fillna(1.0, inplace=True)
        elif 'buy_ratio_1m' in df.columns:
            df['buy_sell_ratio'] = df['buy_ratio_1m'] / (1 - df['buy_ratio_1m']).replace(0, np.nan)
            df['buy_sell_ratio'].fillna(1.0, inplace=True)
        else:
            df['buy_sell_ratio'] = 1.0

    df['recent_high'] = df['high'].rolling(window=price_lookback, min_periods=1).max()

    logging.info("指标预计算完成。")
    return df

def generate_entry_signal(df_row: pd.Series, trader_obj: PaperTrader) -> bool:
    """生成开仓信号"""
    if trader_obj.position != 0:
        return False
    
    conditions_met = []
    
    if ENABLE_OI_TOTAL_INCREASE_ENTRY:
        oi_increase = df_row[f'oi_total_change_pct_{OI_VOLUME_WINDOW_SIZE_ENTRY}'] >= OI_TOTAL_INCREASE_THRESHOLD_ENTRY
        conditions_met.append(oi_increase)
        
    if ENABLE_OI_LONG_INCREASE_ENTRY:
        oi_long_increase = df_row[f'oi_long_change_pct_{OI_VOLUME_WINDOW_SIZE_ENTRY}'] >= OI_LONG_INCREASE_THRESHOLD_ENTRY
        conditions_met.append(oi_long_increase)
        
    if ENABLE_OI_SHORT_DECREASE_ENTRY:
        oi_short_decrease = df_row[f'oi_short_change_pct_{OI_VOLUME_WINDOW_SIZE_ENTRY}'] <= -OI_SHORT_DECREASE_THRESHOLD_ENTRY
        conditions_met.append(oi_short_decrease)
        
    if ENABLE_VOLUME_INCREASE_ENTRY:
        volume_increase = df_row[f'volume_ratio_{OI_VOLUME_WINDOW_SIZE_ENTRY}'] >= VOLUME_INCREASE_MULTIPLIER_ENTRY
        conditions_met.append(volume_increase)
        
    if ENABLE_BUY_SELL_VOLUME_RATIO_ENTRY:
        buy_sell_ratio_met = df_row['buy_sell_ratio'] >= BUY_SELL_VOLUME_RATIO_THRESHOLD_ENTRY
        conditions_met.append(buy_sell_ratio_met)
        
    if ENABLE_PRICE_COOPERATION_ENTRY:
        price_cooperation = df_row['close'] >= df_row['recent_high'] * 0.98
        conditions_met.append(price_cooperation)
    
    if ENTRY_CONDITIONS_COMBINATION == "AND":
        return all(conditions_met) if conditions_met else False
    else:
        return any(conditions_met) if conditions_met else False

def generate_exit_signal(df_row: pd.Series, trader_obj: PaperTrader) -> Tuple[bool, str]:
    """生成平仓信号"""
    if trader_obj.position == 0:
        return False, ""
    
    if ENABLE_STOP_LOSS:
        stop_loss_price = trader_obj.entry_price * (1 - STOP_LOSS_PERCENT)
        if df_row['low'] <= stop_loss_price:
            return True, "STOP_LOSS"
            
    if ENABLE_TAKE_PROFIT:
        take_profit_price = trader_obj.entry_price * (1 + TAKE_PROFIT_PERCENT)
        if df_row['high'] >= take_profit_price:
            return True, "TAKE_PROFIT"
            
    conditions_met = []
    
    if ENABLE_OI_TOTAL_DECREASE_EXIT:
        oi_decrease = df_row[f'oi_total_change_pct_{OI_VOLUME_WINDOW_SIZE_EXIT}'] <= -OI_TOTAL_DECREASE_THRESHOLD_EXIT
        conditions_met.append(oi_decrease)
        
    if ENABLE_OI_LONG_DECREASE_EXIT:
        oi_long_decrease = df_row[f'oi_long_change_pct_{OI_VOLUME_WINDOW_SIZE_EXIT}'] <= -OI_LONG_DECREASE_THRESHOLD_EXIT
        conditions_met.append(oi_long_decrease)
        
    if ENABLE_OU_SHORT_INCREASE_EXIT:
        oi_short_increase = df_row[f'oi_short_change_pct_{OI_VOLUME_WINDOW_SIZE_EXIT}'] >= OI_SHORT_INCREASE_THRESHOLD_EXIT
        conditions_met.append(oi_short_increase)
        
    if ENABLE_VOLUME_DECREASE_EXIT:
        volume_decrease = df_row[f'volume_ratio_{OI_VOLUME_WINDOW_SIZE_EXIT}'] <= VOLUME_DECREASE_MULTIPLIER_EXIT
        conditions_met.append(volume_decrease)
        
    if ENABLE_BUY_SELL_VOLUME_RATIO_EXIT:
        buy_sell_ratio_met = df_row['buy_sell_ratio'] <= BUY_SELL_VOLUME_RATIO_THRESHOLD_EXIT
        conditions_met.append(buy_sell_ratio_met)
    
    if EXIT_CONDITIONS_COMBINATION == "AND":
        signal = all(conditions_met) if conditions_met else False
    else:
        signal = any(conditions_met) if conditions_met else False
    
    return signal, "SIGNAL" if signal else ""

# --- 回测主流程函数 ---
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """加载和预处理数据"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"数据加载成功，共 {len(df)} 行")
        
        if 'create_time' in df.columns:
            df['create_time'] = pd.to_datetime(df['create_time'])
            df.set_index('create_time', inplace=True)
        elif 'minute' in df.columns:
            df['minute'] = pd.to_datetime(df['minute'])
            df.set_index('minute',CEPTIONS = True)
            logging.warning("未找到合适的时间戳列 (create_time 或 minute)，无法进行回测")
            return pd.DataFrame()

        if 'sum_open_interest' in df.columns and 'oi' not in df.columns:
            df['oi'] = df['sum_open_interest']
        
        df.sort_index(inplace=True)
        
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        logging.info(f"日期过滤后剩余 {len(df)} 行数据")
        logging.info(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
        logging.info(f"数据列: {list(df.columns)}")

        target_freq = TARGET_FREQ
        try:
            if not hasattr(df.index, 'freq') or df.index.freq != target_freq:
                logging.info(f"聚合数据到 {target_freq} K线...")
                agg_funcs = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'oi': 'last',
                    'sum_taker_long_short_vol_ratio': 'mean',
                    'taker_buy_volume': 'sum',
                }
                existing_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
                df_agg = df.resample(target_freq).agg(existing_agg_funcs).dropna()
                logging.info(f"聚合后剩余 {len(df_agg)} 行数据")
                df = df_agg
        except ValueError as ve:
            logging.warning(f"聚合数据时发生错误: {ve}，请检查时间戳格式或数据完整性")
            return pd.DataFrame()

        df = precompute_indicators(df, OI_VOLUME_WINDOW_SIZE_ENTRY, OI_VOLUME_WINDOW_SIZE_EXIT, PRICE_COOPERATION_LOOKBACK_ENTRY)
        
        return df
        
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        logging.info("创建示例数据进行演示...")
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq=TARGET_FREQ)
        np.random.seed(42)
        
        price_base = 1.86
        df = pd.DataFrame({
            'open': price_base + np.random.randn(len(dates)).cumsum() * 0.01,
            'high': 0,
            'low': 0,
            'close': 0,
            'volume': np.random.exponential(1000000, len(dates)),
            'oi': 70000000 + np.random.randn(len(dates)).cumsum() * 100000,
            'sum_taker_long_short_vol_ratio': 0.8 + np.random.rand(len(dates)) * 0.4,
            'taker_buy_volume': np.random.exponential(500000, len(dates))
        }, index=dates)
        
        df['high'] = df['open'] + np.random.exponential(0.005, len(dates))
        df['low'] = df['open'] - np.random.exponential(0.005, len(dates))
        df['close'] = df['open'] + np.random.randn(len(dates)) * 0.003
        
        df['oi'] = df['oi'].mask(df['oi'] < 0, 0)
        df['taker_buy_volume'] = df['taker_buy_volume'].mask(df['taker_buy_volume'] < 0, 0)
        
        logging.info(f"示例数据创建完成，共 {len(df)} 行")
        df = precompute_indicators(df, OI_VOLUME_WINDOW_SIZE_ENTRY, OI_VOLUME_WINDOW_SIZE_EXIT, PRICE_COOPERATION_LOOKBACK_ENTRY)
        return df

def calculate_performance_metrics(trader: PaperTrader) -> Dict:
    """计算回测performance指标"""
    if not trader.trade_logs:
        return {}
    
    trades = [log for log in trader.trade_logs if 'net_pnl' in log]
    
    if not trades:
        return {}
    
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['net_pnl'] > 0]
    losing_trades = [t for t in trades if t['net_pnl'] < 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = sum([t['net_pnl'] for t in trades])
    total_return = total_pnl / trader.initial_balance
    
    avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
    
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    balance_history = trader.balance_history
    if balance_history:
        peak = balance_history[0]
        max_drawdown = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
    else:
        max_drawdown = 0
    
    return {
        'net_profit': total_pnl,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_drawdown': max_drawdown,
        'final_balance': trader.balance
    }

def run_backtest():
    """运行回测"""
    logging.info("=" * 60)
    logging.info(f"开始回测 {SYMBOL}")
    logging.info("=" * 60)
    
    df = load_and_preprocess_data(DATA_FILE_PATH)
    if df.empty:
        logging.error("数据为空，无法进行回测")
        return
    
    trader = PaperTrader(INITIAL_BALANCE, TRADING_FEE_RATE, SLIPPAGE_RATE, POSITION_SIZE_TYPE, FIXED_AMOUNT, FIXED_PERCENT)
    
    logging.info(f"\n开始逐K线回测...")
    df_valid = df.dropna()
    if df_valid.empty:
        logging.error("有效数据不足，无法进行回测。请检查数据长度和窗口大小。")
        return

    for i in range(len(df_valid)):
        df_row = df_valid.iloc[i]
        current_timestamp = df_valid.index[i]
        current_price = df_row['close']
        
        exit_signal, exit_reason = generate_exit_signal(df_row, trader)
        if exit_signal:
            if exit_reason == "STOP_LOSS":
                trader.close_long(trader.entry_price * (1 - STOP_LOSS_PERCENT), current_timestamp, exit_reason)
            elif exit_reason == "TAKE_PROFIT":
                trader.close_long(trader.entry_price * (1 + TAKE_PROFIT_PERCENT), current_timestamp, exit_reason)
            else:
                trader.close_long(current_price, current_timestamp, exit_reason)
        
        entry_signal = generate_entry_signal(df_row, trader)
        if entry_signal:
            trader.open_long(current_price, current_timestamp)
        
        trader.update_balance_history(current_price, current_timestamp)
    
    if trader.position == 1:
        final_price = df_valid['close'].iloc[-1]
        final_timestamp = df_valid.index[-1]
        trader.close_long(final_price, final_timestamp, "FINAL")
    
    logging.info("\n" + "=" * 60)
    logging.info("回测结果")
    logging.info("=" * 60)
    
    metrics = calculate_performance_metrics(trader)
    if metrics:
        logging.info(f"净利润: ${metrics['net_profit']:.2f}")
        logging.info(f"总收益率: {metrics['total_return']:.2%}")
        logging.info(f"最终余额: ${metrics['final_balance']:.2f}")
        logging.info(f"总交易次数: {metrics['total_trades']}")
        logging.info(f"盈利交易: {metrics['winning_trades']}")
        logging.info(f"亏损交易: {metrics['losing_trades']}")
        logging.info(f"胜率: {metrics['win_rate']:.2%}")
        logging.info(f"平均盈利: ${metrics['avg_win']:.2f}")
        logging.info(f"平均亏损: ${metrics['avg_loss']:.2f}")
        logging.info(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
        logging.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
    else:
        logging.info("无交易记录")
    
    visualize_results(df, trader, metrics)

def visualize_results(df: pd.DataFrame, trader: PaperTrader, metrics: Dict):
    """可视化回测结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{SYMBOL} 回测结果', fontsize=16)
    
    ax1.plot(df.index, df['close'], label='价格', linewidth=1)
    
    label_open = False
    label_close_signal = False
    label_stop_loss = False
    label_take_profit = False

    for log in trader.trade_logs:
        if log['action'] == 'OPEN_LONG':
            ax1.scatter(log['timestamp'], log['price'], color='green', marker='^', s=100, label='开多' if not label_open else "")
            label_open = True
        elif 'CLOSE_LONG' in log['action']:
            color = 'red'
            marker = 'v'
            label_text = ''
            if 'STOP_LOSS' in log['action']:
                label_text = '止损' if not label_stop_loss else ""
                label_stop_loss = True
            elif 'TAKE_PROFIT' in log['action']:
                label_text = '止盈' if not label_take_profit else ""
                label_take_profit = True
            else:
                color = 'blue'
                label_text = '信号平仓' if not label_close_signal else ""
                label_close_signal = True
            ax1.scatter(log['timestamp'], log['price'], color=color, marker=marker, s=100, label=label_text)
            
    ax1.set_title('价格走势与交易点位')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if trader.balance_history and trader.timestamps:
        ax2.plot(trader.timestamps, trader.balance_history, label='账户余额', color='purple')
        ax2.axhline(y=trader.initial_balance, color='gray', linestyle='--', label='初始资金')
        ax2.set_title('资金曲线')
        ax2.set_ylabel('余额 ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    if 'oi' in df.columns:
        ax3.plot(df.index, df['oi'], label='持仓量(OI)', color='orange')
        ax3.set_title('持仓量变化')
        ax3.set_ylabel('OI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    if 'volume' in df.columns:
        ax4.bar(df.index, df['volume'], alpha=0.6, label='交易量', color='cyan')
        ax4.set_title('交易量')
        ax4.set_ylabel('Volume')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 由于我们不能直接调用 plt.show()，我们将图表保存为文件
    plt.savefig('backtest_results.png')
    logging.info("回测结果图表已保存为 'backtest_results.png'")
    
    logging.info("\n" + "=" * 60)
    logging.info("详细交易记录")
    logging.info("=" * 60)
    for i, log in enumerate(trader.trade_logs, 1):
        if 'net_pnl' in log:
            action_info = log['action'].replace('CLOSE_LONG_', '平仓_')
            logging.info(f"交易 {i}: {log['timestamp']} | {action_info} | "
                         f"入场价: {log.get('entry_price', 0):.4f} | 平仓价: {log['price']:.4f} | "
                         f"盈亏: ${log['net_pnl']:.2f} | 手续费: ${log['fee']:.4f} | "
                         f"当前余额: ${log['balance_after']:.2f}")
        elif log['action'] == 'OPEN_LONG':
            logging.info(f"交易 {i}: {log['timestamp']} | 开多 | "
                         f"价格: {log['price']:.4f} | 数量: {log['position_size']:.4f} | "
                         f"手续费: ${log['fee']:.4f} | 余额前: ${log['balance_before']:.2f} | 余额后: ${trader.balance_history[-1]:.2f} (估算)")

# --- 主程序入口 ---
if __name__ == '__main__':
    logging.info("加密货币 OI & 交易量驱动型多头策略回测工具")
    logging.info("请确保您的数据文件路径正确，并包含必要的列：create_time, open, high, low, close, volume, sum_open_interest, sum_taker_long_short_vol_ratio, taker_buy_volume")
    logging.info("如果没有数据文件，程序将创建示例数据进行演示\n")
    
    run_backtest()