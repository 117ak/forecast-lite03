import random
import pandas as pd

def _fallback(symbol, sample_size, horizon_days, random_seed):
    """
    簡單的預測 fallback 方法
    - 確保 symbol 處理正確（避免 Series 判斷錯誤）
    - 回傳 horizon_days 長度的 baseline 預測
    """
    # 確保 symbol 是字串
    if isinstance(symbol, pd.Series):
        symbol_value = str(symbol.iloc[0])
    else:
        symbol_value = str(symbol)

    # 設定基準價格
    base = 2400.0 if symbol_value.upper() == "XAUUSD" else 1.28

    # 固定亂數種子（保證可重現）
    random.seed(random_seed)

    # 模擬預測結果
    return [base + random.uniform(-5, 5) for _ in range(horizon_days)]


def run_prediction(df, sample_size, horizon_days, random_seed):
    """
    預測主函數
    - df: 使用者上傳的資料
    - sample_size: 取樣大小
    - horizon_days: 預測天數
    - random_seed: 隨機種子
    """
    if df is None or df.empty:
        return _fallback("XAUUSD", sample_size, horizon_days, random_seed)

    try:
        # 嘗試從 df 讀取 symbol 欄位
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "XAUUSD"
    except Exception:
        symbol = "XAUUSD"

    return _fallback(symbol, sample_size, horizon_days, random_seed)
