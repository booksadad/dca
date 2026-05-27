import pandas as pd
import numpy as np

def robust_mad_zscore(series):
    median = series.median()
    mad = (series - median).abs().median()
    return (series - median) / (mad * 1.4826 + 1e-9)

def two_pass_zscore(df, factor_col, peer_group_col='Sector', min_peers=4):
    df_out = df.copy()
    def process_local(x):
        if len(x) < min_peers: return pd.Series(np.nan, index=x.index)
        return robust_mad_zscore(x.clip(x.quantile(0.05), x.quantile(0.95)))

    df_out['z1_local'] = df_out.groupby(peer_group_col)[factor_col].transform(process_local)
    global_z = robust_mad_zscore(df_out[factor_col].clip(df_out[factor_col].quantile(0.05), df_out[factor_col].quantile(0.95)))
    df_out['z1_local'] = df_out['z1_local'].fillna(global_z)
    return robust_mad_zscore(df_out['z1_local']).fillna(0)

def calculate_alpha_decay(signal_value, days_passed, half_life_days):
    return signal_value * np.exp(-(np.log(2) / half_life_days) * days_passed) if days_passed > 0 else signal_value

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / (loss + 1e-9))))

def check_doi_risk(rsi_val):
    if rsi_val > 75: return "🔴 เสี่ยงดอย (Overbought)"
    elif rsi_val < 30: return "🟢 โซนเก็บของ (Oversold)"
    return "🟡 กลางๆ (Neutral)"

def find_sr_levels(series):
    return f"รับ: {series.quantile(0.2):.2f} / ต้าน: {series.quantile(0.8):.2f}"
