import importlib.util, os, random, datetime, math

HERE = os.path.dirname(__file__)
WEIGHTS_PATHS = [
    os.path.join(HERE, "weights", "lstm_model.h5"),
    os.path.join(HERE, "originals", "lstm_model.h5"),
    os.path.join(HERE, "lstm_model.h5"),
]

def _try_load_keras():
    try:
        from tensorflow import keras
    except Exception:
        return None, None
    weights = next((p for p in WEIGHTS_PATHS if os.path.exists(p)), None)
    if not weights: return None, None
    try:
        model = keras.models.load_model(weights, compile=False)
    except Exception:
        return None, None
    def preprocess(series, lookback=48):
        import numpy as np
        x = np.array(series[-lookback:], dtype="float32").reshape(1, lookback, 1)
        return x
    return model, preprocess

def _fallback(symbol, sample_size, horizon_days, seed):
    if seed is not None: random.seed(seed+23)
    anchor = 2400.0 if symbol=="XAUUSD" else 1.28
    vol = (28.0 if symbol=="XAUUSD" else 0.0045) / math.sqrt(sample_size)
    today = datetime.date.today(); level = anchor; out=[]
    for i in range(horizon_days):
        shock = random.gauss(0.0, vol)
        level = max(0.0, level + (anchor - level)*0.10 + shock)
        t = datetime.datetime.combine(today+datetime.timedelta(days=i+1), datetime.time()).isoformat()+"Z"
        out.append((t, level))
    return out

def run_prediction(symbol: str, sample_size: int, horizon_days: int, random_seed: int|None=None):
    # 若 symbol 不支援 fallback
    if symbol != "XAUUSD":
        return _fallback(symbol, sample_size, horizon_days, random_seed)

    # 嘗試 keras 權重
    model, preprocess = _try_load_keras()
    if model is not None and preprocess is not None:
        import numpy as np
        if random_seed is not None: np.random.seed(random_seed); random.seed(random_seed)
        anchor = 2400.0
        sigma = (28.0 / (sample_size ** 0.5)) * 0.2
        series = [anchor + random.gauss(0.0, sigma) for _ in range(48)]
        today = datetime.date.today(); out=[]
        level = series[-1]
        for i in range(horizon_days):
            x = preprocess(series, lookback=48)
            yhat = model.predict(x, verbose=0)
            val = float(yhat.reshape(-1)[0])
            if 0.0 <= val <= 1.0:
                val = anchor*(0.98 + 0.04*val)
            if abs(val) < 10:
                level = max(0.0, level + val)
            else:
                level = max(0.0, val)
            series.append(level)
            t = datetime.datetime.combine(today+datetime.timedelta(days=i+1), datetime.time()).isoformat()+"Z"
            out.append((t, level))
        return out

    # fallback
    return _fallback(symbol, sample_size, horizon_days, random_seed)
