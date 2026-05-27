import numpy as np
from scipy.signal import argrelextrema

def calc_zscore(series): 
    if series.std() == 0: return series - series.mean()
    return (series - series.mean()) / series.std()

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_doi_risk(rsi_val):
    if rsi_val > 75: return 'ดอย (ซื้อระวัง)'
    elif rsi_val < 30: return 'ของถูก (เก็บสะสม)'
    return 'ปกติ'

def find_sr_levels(series):
    try:
        current_price = series.iloc[-1]
        local_mins = series.iloc[argrelextrema(series.values, np.less_equal, order=5)[0]]
        local_maxs = series.iloc[argrelextrema(series.values, np.greater_equal, order=5)[0]]
        supports = local_mins[local_mins < current_price].sort_values(ascending=False).unique()
        resistances = local_maxs[local_maxs > current_price].sort_values(ascending=True).unique()
        s1 = supports[0] if len(supports) > 0 else current_price * 0.95
        s2 = supports[1] if len(supports) > 1 else s1 * 0.95
        r1 = resistances[0] if len(resistances) > 0 else current_price * 1.05
        return f"รับ: {s1:.1f}, {s2:.1f} | ต้าน: {r1:.1f}"
    except: return "-"
