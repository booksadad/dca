import pandas as pd
import numpy as np

def calc_zscore(series):
    if series.std() == 0: return pd.Series(0, index=series.index)
    return (series - series.mean()) / series.std()

def calculate_smart_weights(df, max_w=40.0, hold_w=5.0):
    w = pd.Series(0.0, index=df.index)
    sell_idx = df[(df['Rank'] > 15) & (df['Price'] < df['SMA_200'])].index
    hold_idx = df[(df['Rank'] > 15) & (df['Price'] >= df['SMA_200'])].index
    elite_idx = df[df['Rank'] <= 15].index
    
    w.loc[sell_idx] = 0.0
    w.loc[hold_idx] = hold_w
    
    rem_w = 100.0 - w.sum()
    if len(elite_idx) > 0 and rem_w > 0:
        scores = df.loc[elite_idx, 'Alpha_Score']
        exp_s = np.exp(scores)
        elite_w = (exp_s / exp_s.sum()) * rem_w
        for _ in range(10):
            over = elite_w > max_w
            if not over.any(): break
            excess = (elite_w[over] - max_w).sum()
            elite_w[over] = max_w
            free = ~over
            if free.any(): elite_w[free] += excess * (elite_w[free] / elite_w[free].sum())
            else: 
                elite_w += excess / len(elite_w)
                break
        w.loc[elite_idx] = elite_w
    return w

def allocate_v21_fixed(df_def, total_budget, min_order):
    eligible = df_def[df_def['Deficit'] > 0].copy()
    allocs = pd.Series(0.0, index=df_def.index)
    if eligible.empty: return allocs
    remaining = total_budget
    while not eligible.empty and remaining >= min_order:
        total_def = eligible['Deficit'].sum()
        temp = (eligible['Deficit'] / total_def) * remaining
        below = temp[(temp > 0) & (temp < min_order)]
        if below.empty: 
            allocs[temp.index] = temp
            break
        eligible = eligible.drop(below.idxmin())
    current_sum = allocs.sum()
    if 0 < current_sum < total_budget:
        gap = total_budget - current_sum
        allocs[allocs.idxmax()] += gap
    return allocs.round(2)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_doi_risk(rsi):
    if rsi >= 75: return '🌋 ดอย (ซื้อระวัง)'
    elif rsi >= 60: return '🔥 ซิ่ง (กราฟพุ่ง)'
    elif rsi <= 40: return '💎 ของถูก (เก็บสะสม)'
    else: return '⚖️ ราคากลางๆ'
    
def calculate_support_resistance(price_series, window=20):
    # ใช้ราคาปิด 20 วันทำการล่าสุด (ประมาณ 1 เดือน) หาจุดต่ำสุด (แนวรับ) และสูงสุด (แนวต้าน)
    support = price_series.tail(window).min()
    resistance = price_series.tail(window).max()
    return support, resistance