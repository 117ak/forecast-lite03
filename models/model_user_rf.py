# models/model_user_rf.py

import random
import pandas as pd

def _fallback(symbol, sample_size, horizon_days, random_seed):
    """
    修正過的 fallback 方法
    確保 symbol 是單一字串，不會整個列 (Series) 去比對
    """

    # 如果 symbol 是 DataFrame 或 Series，取第一筆元素
    try:
        if isinstance(symbol, (pd.Series, pd.DataFrame)):
            # 如果是 Series
            symbol_value = str(symbol.iloc[0])
        else:
            symbol_value = str(symbol)
    except Exception:
        # 如果讀不到，就預設 XAUUSD
        symbol_value = "XAUUSD"

    # 比對字串
    if symbol_value.upper() == "XAUUSD":
        base = 2400.0
    else:
        base = 1.28

    # 設定 random seed
    random.seed(random_seed)

    # 回傳一組預測值（horizon_days 長度）
    points = []
    for _ in range(horizon_days):
        # 這裡你可以調整隨機範圍
        points.append(base + random.uniform(-5, 5))
    return points

def run_prediction(df, sample_size, horizon_days, random_seed):
    """
    main prediction function
    如果傳進來 df 是空或不存在，就 fallback
    否則用 symbol 值做 fallback 預測
    """

    if df is None or df.empty:
        return _fallback("XAUUSD", sample_size, horizon_days, random_seed)

    # 嘗試從 df 讀 symbol 欄位
    if "symbol" in df.columns:
        # 擷取第一列 symbol
        symbol = df["symbol"].iloc[0]
    else:
        symbol = "XAUUSD"

    return _fallback(symbol, sample_size, horizon_days, random_seed)

