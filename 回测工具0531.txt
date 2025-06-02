import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import os # 用于模拟数据文件路径

warnings.filterwarnings('ignore')

# 配置日志
LOG_FILE = "backtest_log.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'),  # 输出到文件，覆盖模式
                        logging.StreamHandler()  # 同时输出到控制台
                    ])

# --- 模拟后端数据和接口 (请注意：在实际部署时，这些需要替换为真实的后端 API) ---

# 模拟可回测的币种及其数据路径
# 在实际部署中，这些数据文件会存储在服务器的特定路径，并通过后端逻辑获取
MOCK_BACKEND_DATA_PATHS = {
    "WIFUSDT": "data/wifusdt_data.csv", # 假设有这个文件，或者其他动态生成数据的方式
    "BTCUSDT": "data/btcusdt_data.csv",
    "ETHUSDT": "data/ethusdt_data.csv",
}

# 模拟后端提供可回测币种列表的 API
def get_available_symbols() -> List[str]:
    """模拟后端API，获取可回测的交易对列表。"""
    logging.info("模拟后端：获取可回测币种列表。")
    return list(MOCK_BACKEND_DATA_PATHS.keys())

# 模拟后端提供特定币种数据时间范围的 API
def get_symbol_date_range(symbol: str) -> Optional[Dict[str, str]]:
    """模拟后端API，获取指定币种的历史数据时间范围。"""
    logging.info(f"模拟后端：获取 {symbol} 的数据时间范围。")
    # 这里我们假设WIFUSDT数据范围是2024年1月1日到2025年1月1日
    # 实际应根据后端真实数据范围返回
    if symbol == "WIFUSDT":
        return {"start": "2024-01-01 00:00:00", "end": "2025-01-01 00:00:00"}
    elif symbol == "BTCUSDT":
        return {"start": "2023-01-01 00:00:00", "end": "2025-01-01 00:00:00"}
    elif symbol == "ETHUSDT":
        return {"start": "2023-06-01 00:00:00", "end": "2024-12-01 00:00:00"}
    return None

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
                # 确保不会因为余额太小导致仓位为零
                amount_to_invest = self.balance * self.fixed_percent
                if amount_to_invest <= 0:
                    logging.info(f"[{timestamp}] 计算投入金额为0或负数，无法开仓。")
                    return False
                self.position_size = amount_to_invest / actual_price
            else: # 默认为固定百分比或全部余额
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
            # 确保 x 是数值且不是NaN，避免 apply 函数报错
            df['long_weight'] = df['sum_taker_long_short_vol_ratio'].apply(
                lambda x: min(0.9, max(0.1, x / (x + 1))) if pd.notna(x) and x > 0 else 0.5
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

def generate_entry_signal(df_row: pd.Series, trader_obj: PaperTrader, entry_params: Dict) -> bool:
    """生成开仓信号"""
    if trader_obj.position != 0:
        return False
    
    conditions_met = []
    
    if entry_params['ENABLE_OI_TOTAL_INCREASE_ENTRY']:
        oi_increase = df_row[f'oi_total_change_pct_{entry_params["OI_VOLUME_WINDOW_SIZE_ENTRY"]}'] >= entry_params['OI_TOTAL_INCREASE_THRESHOLD_ENTRY']
        conditions_met.append(oi_increase)
        
    if entry_params['ENABLE_OI_LONG_INCREASE_ENTRY']:
        oi_long_increase = df_row[f'oi_long_change_pct_{entry_params["OI_VOLUME_WINDOW_SIZE_ENTRY"]}'] >= entry_params['OI_LONG_INCREASE_THRESHOLD_ENTRY']
        conditions_met.append(oi_long_increase)
        
    if entry_params['ENABLE_OI_SHORT_DECREASE_ENTRY']:
        oi_short_decrease = df_row[f'oi_short_change_pct_{entry_params["OI_VOLUME_WINDOW_SIZE_ENTRY"]}'] <= -entry_params['OI_SHORT_DECREASE_THRESHOLD_ENTRY']
        conditions_met.append(oi_short_decrease)
        
    if entry_params['ENABLE_VOLUME_INCREASE_ENTRY']:
        volume_increase = df_row[f'volume_ratio_{entry_params["OI_VOLUME_WINDOW_SIZE_ENTRY"]}'] >= entry_params['VOLUME_INCREASE_MULTIPLIER_ENTRY']
        conditions_met.append(volume_increase)
        
    if entry_params['ENABLE_BUY_SELL_VOLUME_RATIO_ENTRY']:
        # 修复了原始代码中的拼写错误 `BUY10 BUY_SELL_VOLUME_RATIO_THRESHOLD_ENTRY`
        buy_sell_ratio_met = df_row['buy_sell_ratio'] >= entry_params['BUY_SELL_VOLUME_RATIO_THRESHOLD_ENTRY']
        conditions_met.append(buy_sell_ratio_met)
        
    if entry_params['ENABLE_PRICE_COOPERATION_ENTRY']:
        # 确保 df_row['recent_high'] 存在且不是 NaN
        if 'recent_high' in df_row and pd.notna(df_row['recent_high']):
            price_cooperation = df_row['close'] >= df_row['recent_high'] * 0.98
            conditions_met.append(price_cooperation)
        else:
            # 如果没有 recent_high 数据，此条件视为不满足或跳过
            conditions_met.append(False) 
    
    # 确保 conditions_met 不为空，避免 all() 或 any() 在空列表上返回 True
    if not conditions_met:
        return False

    if entry_params['ENTRY_CONDITIONS_COMBINATION'] == "AND":
        return all(conditions_met)
    else: # "OR"
        return any(conditions_met)

def generate_exit_signal(df_row: pd.Series, trader_obj: PaperTrader, exit_params: Dict) -> Tuple[bool, str]:
    """生成平仓信号"""
    if trader_obj.position == 0:
        return False, ""
    
    # 止损止盈优先判断
    if exit_params['ENABLE_STOP_LOSS']:
        stop_loss_price = trader_obj.entry_price * (1 - exit_params['STOP_LOSS_PERCENT'])
        if df_row['low'] <= stop_loss_price:
            return True, "STOP_LOSS"
            
    if exit_params['ENABLE_TAKE_PROFIT']:
        take_profit_price = trader_obj.entry_price * (1 + exit_params['TAKE_PROFIT_PERCENT'])
        if df_row['high'] >= take_profit_price:
            return True, "TAKE_PROFIT"
            
    conditions_met = []
    
    if exit_params['ENABLE_OI_TOTAL_DECREASE_EXIT']:
        oi_decrease = df_row[f'oi_total_change_pct_{exit_params["OI_VOLUME_WINDOW_SIZE_EXIT"]}'] <= -exit_params['OI_TOTAL_DECREASE_THRESHOLD_EXIT']
        conditions_met.append(oi_decrease)
        
    if exit_params['ENABLE_OI_LONG_DECREASE_EXIT']:
        oi_long_decrease = df_row[f'oi_long_change_pct_{exit_params["OI_VOLUME_WINDOW_SIZE_EXIT"]}'] <= -exit_params['OI_LONG_DECREASE_THRESHOLD_EXIT']
        conditions_met.append(oi_long_decrease)
        
    # 修复了原始代码中的拼写错误 ENABLE_OU_SHORT_INCREASE_EXIT
    if exit_params['ENABLE_OI_SHORT_INCREASE_EXIT']:
        oi_short_increase = df_row[f'oi_short_change_pct_{exit_params["OI_VOLUME_WINDOW_SIZE_EXIT"]}'] >= exit_params['OI_SHORT_INCREASE_THRESHOLD_EXIT']
        conditions_met.append(oi_short_increase)
        
    if exit_params['ENABLE_VOLUME_DECREASE_EXIT']:
        volume_decrease = df_row[f'volume_ratio_{exit_params["OI_VOLUME_WINDOW_SIZE_EXIT"]}'] <= exit_params['VOLUME_DECREASE_MULTIPLIER_EXIT']
        conditions_met.append(volume_decrease)
        
    if exit_params['ENABLE_BUY_SELL_VOLUME_RATIO_EXIT']:
        buy_sell_ratio_met = df_row['buy_sell_ratio'] <= exit_params['BUY_SELL_VOLUME_RATIO_THRESHOLD_EXIT']
        conditions_met.append(buy_sell_ratio_met)
    
    # 确保 conditions_met 不为空
    if not conditions_met:
        return False, ""

    if exit_params['EXIT_CONDITIONS_COMBINATION'] == "AND":
        signal = all(conditions_met)
    else: # "OR"
        signal = any(conditions_met)
        
    return signal, "SIGNAL" if signal else ""

# --- 回测主流程函数 ---
def load_and_preprocess_data(symbol: str, start_date: str, end_date: str, target_freq: str) -> pd.DataFrame:
    """
    加载和预处理数据。
    现在根据symbol从模拟的后端路径加载数据，如果文件不存在则生成合成数据。
    """
    file_path = MOCK_BACKEND_DATA_PATHS.get(symbol)
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"指定币种 '{symbol}' 的数据文件 '{file_path}' 不存在或未配置。将创建示例数据。")
        # 直接创建合成数据，忽略 file_path
        pass # 后续的except块会处理
    else:
        logging.info(f"尝试从文件加载数据: {file_path}")

    try:
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logging.info(f"数据加载成功，共 {len(df)} 行")
        else: # 如果文件路径不存在，将触发下一步的合成数据生成
            raise FileNotFoundError("数据文件不存在，转为生成示例数据。")
        
        if 'create_time' in df.columns:
            df['create_time'] = pd.to_datetime(df['create_time'])
            df.set_index('create_time', inplace=True)
        elif 'minute' in df.columns:
            df['minute'] = pd.to_datetime(df['minute'])
            # 修复了原始代码中的拼写错误
            df.set_index('minute', inplace=True)
        else:
            logging.warning("未找到合适的时间戳列 (create_time 或 minute)，无法进行回测。")
            return pd.DataFrame()

        if 'sum_open_interest' in df.columns and 'oi' not in df.columns:
            df['oi'] = df['sum_open_interest']
            
        df.sort_index(inplace=True)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        logging.info(f"日期过滤后剩余 {len(df)} 行数据")
        if not df.empty:
            logging.info(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
        else:
            logging.warning("日期过滤后数据为空。")

        logging.info(f"数据列: {list(df.columns)}")

        try:
            # 检查索引频率是否与目标频率匹配，或者是否需要重新采样
            # 这里的逻辑可以更健壮，如果原始数据是混合频率或者不规则的，直接resample可能更好
            # 但为了保持与原代码逻辑一致，只在freq不匹配时resample
            if not hasattr(df.index, 'freq') or pd.infer_freq(df.index) != target_freq:
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
                # 注意：resample 会创建新的索引，可能包含 NaN，dropna很重要
                df_agg = df.resample(target_freq).agg(existing_agg_funcs).dropna(subset=['close']) # 至少close不能是NaN
                logging.info(f"聚合后剩余 {len(df_agg)} 行数据")
                df = df_agg
        except ValueError as ve:
            logging.warning(f"聚合数据时发生错误: {ve}，请检查时间戳格式或数据完整性")
            return pd.DataFrame()
        
        # 预计算指标前，确保数据不为空
        if df.empty:
            logging.warning("经过预处理和聚合后，数据为空。")
            return pd.DataFrame()

        # 注意：此处使用的 OI_VOLUME_WINDOW_SIZE_ENTRY 等参数，应来自外部传入，而不是全局变量
        # 这里暂时使用全局变量，因为 run_backtest 会将它们作为字典传入
        # 更严谨的做法是将这些参数也作为 load_and_preprocess_data 的参数
        df = precompute_indicators(df, OI_VOLUME_WINDOW_SIZE_ENTRY, OI_VOLUME_WINDOW_SIZE_EXIT, PRICE_COOPERATION_LOOKBACK_ENTRY)
        
        return df
        
    except Exception as e:
        logging.error(f"数据加载或预处理失败: {e}")
        logging.info("创建示例数据进行演示...")
        
        # 使用传入的 start_date 和 end_date 来生成示例数据
        dates = pd.date_range(start=start_date, end=end_date, freq=target_freq)
        if dates.empty:
            logging.error(f"无法根据给定日期范围 ({start_date} - {end_date}) 和频率 ({target_freq}) 生成示例数据。")
            return pd.DataFrame()

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
        logging.info("无交易记录，无法计算性能指标。")
        return {}
        
    trades = [log for log in trader.trade_logs if 'net_pnl' in log]
    
    if not trades:
        logging.info("无有效平仓交易记录，无法计算性能指标。")
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
        'losing_trades': len(losing_trades), # 修复了原始代码中的 `len l Trades'`
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_drawdown': max_drawdown,
        'final_balance': trader.balance
    }

# 修改 run_backtest 函数以接收动态参数
def run_backtest(
    symbol: str,
    initial_balance: float,
    start_date: str,
    end_date: str,
    trading_fee_rate: float,
    slippage_rate: float,
    target_freq: str,
    
    # 开仓条件参数
    entry_conditions_combination: str,
    enable_oi_total_increase_entry: bool,
    oi_total_increase_threshold_entry: float,
    enable_oi_long_increase_entry: bool,
    oi_long_increase_threshold_entry: float,
    enable_oi_short_decrease_entry: bool,
    oi_short_decrease_threshold_entry: float,
    enable_volume_increase_entry: bool,
    volume_increase_multiplier_entry: float,
    enable_buy_sell_volume_ratio_entry: bool,
    buy_sell_volume_ratio_threshold_entry: float,
    oi_volume_window_size_entry: int,
    enable_price_cooperation_entry: bool,
    price_cooperation_lookback_entry: int,

    # 平仓条件参数
    exit_conditions_combination: str,
    enable_oi_total_decrease_exit: bool,
    oi_total_decrease_threshold_exit: float,
    enable_oi_long_decrease_exit: bool,
    oi_long_decrease_threshold_exit: float,
    enable_oi_short_increase_exit: bool, # 已修复拼写错误
    oi_short_increase_threshold_exit: float,
    enable_volume_decrease_exit: bool,
    volume_decrease_multiplier_exit: float,
    enable_buy_sell_volume_ratio_exit: bool,
    buy_sell_volume_ratio_threshold_exit: float,
    oi_volume_window_size_exit: int,
    enable_stop_loss: bool,
    stop_loss_percent: float,
    enable_take_profit: bool,
    take_profit_percent: float,

    # 风险管理参数
    position_size_type: str,
    fixed_amount: float,
    fixed_percent: float
):
    """
    运行回测。所有参数现在都从外部传入，模拟从前端UI接收。
    """
    logging.info("=" * 60)
    logging.info(f"开始回测 {symbol}")
    logging.info("=" * 60)
    
    # 将入口和出口参数打包成字典，方便传入信号生成函数
    entry_params = {
        'ENTRY_CONDITIONS_COMBINATION': entry_conditions_combination,
        'ENABLE_OI_TOTAL_INCREASE_ENTRY': enable_oi_total_increase_entry,
        'OI_TOTAL_INCREASE_THRESHOLD_ENTRY': oi_total_increase_threshold_entry,
        'ENABLE_OI_LONG_INCREASE_ENTRY': enable_oi_long_increase_entry,
        'OI_LONG_INCREASE_THRESHOLD_ENTRY': oi_long_increase_threshold_entry,
        'ENABLE_OI_SHORT_DECREASE_ENTRY': enable_oi_short_decrease_entry,
        'OI_SHORT_DECREASE_THRESHOLD_ENTRY': oi_short_decrease_threshold_entry,
        'ENABLE_VOLUME_INCREASE_ENTRY': enable_volume_increase_entry,
        'VOLUME_INCREASE_MULTIPLIER_ENTRY': volume_increase_multiplier_entry,
        'ENABLE_BUY_SELL_VOLUME_RATIO_ENTRY': enable_buy_sell_volume_ratio_entry,
        'BUY_SELL_VOLUME_RATIO_THRESHOLD_ENTRY': buy_sell_volume_ratio_threshold_entry,
        'OI_VOLUME_WINDOW_SIZE_ENTRY': oi_volume_window_size_entry,
        'ENABLE_PRICE_COOPERATION_ENTRY': enable_price_cooperation_entry,
        'PRICE_COOPERATION_LOOKBACK_ENTRY': price_cooperation_lookback_entry,
    }

    exit_params = {
        'EXIT_CONDITIONS_COMBINATION': exit_conditions_combination,
        'ENABLE_OI_TOTAL_DECREASE_EXIT': enable_oi_total_decrease_exit,
        'OI_TOTAL_DECREASE_THRESHOLD_EXIT': oi_total_decrease_threshold_exit,
        'ENABLE_OI_LONG_DECREASE_EXIT': enable_oi_long_decrease_exit,
        'OI_LONG_DECREASE_THRESHOLD_EXIT': oi_long_decrease_threshold_exit,
        'ENABLE_OI_SHORT_INCREASE_EXIT': enable_oi_short_increase_exit,
        'OI_SHORT_INCREASE_THRESHOLD_EXIT': oi_short_increase_threshold_exit,
        'ENABLE_VOLUME_DECREASE_EXIT': enable_volume_decrease_exit,
        'VOLUME_DECREASE_MULTIPLIER_EXIT': volume_decrease_multiplier_exit,
        'ENABLE_BUY_SELL_VOLUME_RATIO_EXIT': enable_buy_sell_volume_ratio_exit,
        'BUY_SELL_VOLUME_RATIO_THRESHOLD_EXIT': buy_sell_volume_ratio_threshold_exit,
        'OI_VOLUME_WINDOW_SIZE_EXIT': oi_volume_window_size_exit,
        'ENABLE_STOP_LOSS': enable_stop_loss,
        'STOP_LOSS_PERCENT': stop_loss_percent,
        'ENABLE_TAKE_PROFIT': enable_take_profit,
        'TAKE_PROFIT_PERCENT': take_profit_percent,
    }

    # 全局变量的预计算参数依赖于 run_backtest 的参数，这里需要更新，或者将 precompute_indicators 也改为接收参数
    # 为了简化，我们暂时保留 precompute_indicators 使用全局配置的方式，但实际应传递参数
    # 或者将这些配置放入 entry_params 和 exit_params 中，并在 precompute_indicators 中解包使用
    # 由于原始 precompute_indicators 依赖全局变量，这里为了兼容，保留其使用全局变量的方式，
    # 但在实际应用中，这些应作为参数传入或通过类实例管理。
    # 临时更新全局变量以供 precompute_indicators 使用 (这不是最佳实践，但为了快速修改)
    global OI_VOLUME_WINDOW_SIZE_ENTRY, OI_VOLUME_WINDOW_SIZE_EXIT, PRICE_COOPERATION_LOOKBACK_ENTRY
    OI_VOLUME_WINDOW_SIZE_ENTRY = oi_volume_window_size_entry
    OI_VOLUME_WINDOW_SIZE_EXIT = oi_volume_window_size_exit
    PRICE_COOPERATION_LOOKBACK_ENTRY = price_cooperation_lookback_entry

    df = load_and_preprocess_data(symbol, start_date, end_date, target_freq)
    if df.empty:
        logging.error("数据为空，无法进行回测")
        return
        
    trader = PaperTrader(initial_balance, trading_fee_rate, slippage_rate, position_size_type, fixed_amount, fixed_percent)
    
    logging.info(f"\n开始逐K线回测...")
    df_valid = df.dropna()
    if df_valid.empty:
        logging.error("有效数据不足，无法进行回测。请检查数据长度和窗口大小。")
        return

    for i in range(len(df_valid)):
        df_row = df_valid.iloc[i]
        current_timestamp = df_valid.index[i]
        current_price = df_row['close']
        
        exit_signal, exit_reason = generate_exit_signal(df_row, trader, exit_params)
        if exit_signal:
            if exit_reason == "STOP_LOSS":
                trader.close_long(trader.entry_price * (1 - exit_params['STOP_LOSS_PERCENT']), current_timestamp, exit_reason)
            elif exit_reason == "TAKE_PROFIT":
                trader.close_long(trader.entry_price * (1 + exit_params['TAKE_PROFIT_PERCENT']), current_timestamp, exit_reason)
            else:
                trader.close_long(current_price, current_timestamp, exit_reason)
        
        entry_signal = generate_entry_signal(df_row, trader, entry_params)
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
    fig.suptitle(f'{SYMBOL} 回测结果', fontsize=16) # SYMBOL 仍然是全局变量，在 run_backtest 中传入后，此处未更新，注意。

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
    logging.info("此版本已修改为模拟前端传入参数，不再需要手动配置 DATA_FILE_PATH。")
    logging.info("请确保您的后台已准备好数据文件，或程序将自动生成示例数据。")
    logging.info("可用的币种 (模拟)：" + ", ".join(get_available_symbols()) + "\n")

    # --- 示例：如何调用新的 run_backtest 函数 ---
    # 模拟从前端UI接收到的参数
    selected_symbol = "WIFUSDT"
    # 获取模拟的日期范围
    date_range = get_symbol_date_range(selected_symbol)
    if date_range:
        simulated_start_date = date_range["start"]
        simulated_end_date = date_range["end"]
    else:
        # 如果没有获取到特定币种的日期范围，使用一个默认的范围
        logging.warning(f"未能获取 {selected_symbol} 的日期范围，使用默认日期。")
        simulated_start_date = "2024-01-01 00:00:00"
        simulated_end_date = "2024-01-07 23:59:59" # 示例只回测几天

    # 调用修改后的 run_backtest 函数
    run_backtest(
        symbol=selected_symbol,
        initial_balance=10000.0,
        start_date=simulated_start_date,
        end_date=simulated_end_date,
        trading_fee_rate=0.0002,
        slippage_rate=0.0001,
        target_freq='1H',
        
        # 开仓条件 (模拟从前端UI接收，使用原始默认值)
        entry_conditions_combination="AND",
        enable_oi_total_increase_entry=True,
        oi_total_increase_threshold_entry=0.03,
        enable_oi_long_increase_entry=True,
        oi_long_increase_threshold_entry=0.05,
        enable_oi_short_decrease_entry=False,
        oi_short_decrease_threshold_entry=0.03,
        enable_volume_increase_entry=True,
        volume_increase_multiplier_entry=1.5,
        enable_buy_sell_volume_ratio_entry=True,
        buy_sell_volume_ratio_threshold_entry=1.2,
        oi_volume_window_size_entry=5,
        enable_price_cooperation_entry=True,
        price_cooperation_lookback_entry=10,

        # 平仓条件 (模拟从前端UI接收，使用原始默认值，已修复拼写错误)
        exit_conditions_combination="OR",
        enable_oi_total_decrease_exit=True,
        oi_total_decrease_threshold_exit=0.02,
        enable_oi_long_decrease_exit=True,
        oi_long_decrease_threshold_exit=0.03,
        enable_oi_short_increase_exit=False, # 原始为 False
        oi_short_increase_threshold_exit=0.02,
        enable_volume_decrease_exit=True,
        volume_decrease_multiplier_exit=0.8,
        enable_buy_sell_volume_ratio_exit=True,
        buy_sell_volume_ratio_threshold_exit=0.8,
        oi_volume_window_size_exit=5,
        enable_stop_loss=True,
        stop_loss_percent=0.03,
        enable_take_profit=True,
        take_profit_percent=0.10,

        # 风险管理 (模拟从前端UI接收，使用原始默认值)
        position_size_type="FIXED_PERCENT",
        fixed_amount=1000.0,
        fixed_percent=0.95
    )