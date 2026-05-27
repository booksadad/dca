import pandas as pd
import numpy as np

def robust_mad_zscore(series):
    """คำนวณ Z-Score ด้วย MAD (ทนทานต่อ Outlier ทุกสภาวะตลาด)"""
    median = series.median()
    mad = (series - median).abs().median()
    mad_scaled = mad * 1.4826 + 1e-9
    return (series - median) / mad_scaled

def two_pass_zscore(df, factor_col, peer_group_col='Sector', min_peers=4):
    """
    Layer 2.1: Institutional Two-pass Sector-neutral Z-score
    - Pass 1: Local MAD Z-score (เทียบในกลุ่ม) + Local Winsorization
    - Fallback: ถ้ากลุ่มเล็กไป (N < min_peers) ให้ใช้ Global MAD ป้องกันค่าเพี้ยน
    - Pass 2: Global MAD Z-score (ปรับสเกลข้ามตลาด)
    """
    df_out = df.copy()
    
    # 1. Local Winsorize & Pass 1 Z-Score
    def process_local(x):
        if len(x) < min_peers:
            return pd.Series(np.nan, index=x.index) # ส่ง NaN ไปรอ Fallback
        
        # Winsorize ภายใน Sector (5th - 95th Percentile)
        lower, upper = x.quantile(0.05), x.quantile(0.95)
        x_clipped = x.clip(lower, upper)
        
        return robust_mad_zscore(x_clipped)

    # คำนวณ Z1 แยกตามกลุ่มอุตสาหกรรม
    df_out['z1_local'] = df_out.groupby(peer_group_col)[factor_col].transform(process_local)
    
    # 2. Fallback Mechanism (เตรียม Global Z-Score สำหรับกลุ่มที่หุ้นน้อย)
    global_lower, global_upper = df_out[factor_col].quantile(0.05), df_out[factor_col].quantile(0.95)
    global_clipped = df_out[factor_col].clip(global_lower, global_upper)
    global_z = robust_mad_zscore(global_clipped)
    
    # อุดรอยรั่วกลุ่มเล็กด้วย Global Z-Score
    df_out['z1_local'] = df_out['z1_local'].fillna(global_z)
    
    # 3. Pass 2: Cross-universe MAD Z-score
    # นำคะแนน Z1 มาหาค่า MAD อีกรอบเพื่อ Normalize ข้ามกลุ่มอย่างยุติธรรมที่สุด
    df_out['z2_final'] = robust_mad_zscore(df_out['z1_local'])
    
    return df_out['z2_final'].fillna(0)

def calculate_alpha_decay(signal_value, days_passed, half_life_days):
    """
    Layer 2.3: Alpha Decay Model
    (เตรียมพร้อมอัปเกรดเป็น Dynamic Decay จากค่า VIX ในอนาคต)
    """
    if days_passed == 0:
        return signal_value
        
    decay_constant = np.log(2) / half_life_days
    decay_weight = np.exp(-decay_constant * days_passed)
    
    return signal_value * decay_weight

def calculate_rsi(series, period=14):
    """Layer 5: Risk Overlay (ไม่นำไปรวมใน Alpha)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def check_doi_risk(rsi_val):
    if rsi_val > 75: return "🔴 เสี่ยงดอย (Overbought)"
    elif rsi_val < 30: return "🟢 โซนเก็บของ (Oversold)"
    return "🟡 กลางๆ (Neutral)"

def find_sr_levels(series):
    """Statistical Distribution Bands"""
    return f"รับ: {series.quantile(0.2):.2f} / ต้าน: {series.quantile(0.8):.2f}"
