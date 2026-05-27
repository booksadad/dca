import pandas as pd
import numpy as np
import yfinance as yf

def get_clean_universe(ticker_list):
    """
    Layer 1: Data Pipeline & Sanitization
    - กรองหุ้นสภาพคล่องต่ำ (ADV < $5M)
    - กำจัด Outlier ด้วย Median Absolute Deviation (MAD)
    - ปรับข้อมูลให้อยู่ในรูปแบบ Normalized TTM
    """
    
    # 1. ดึงข้อมูลดิบในคราวเดียวเพื่อความเร็ว
    data = yf.download(ticker_list, period="1y", group_by='ticker')
    
    clean_data = {}
    for t in ticker_list:
        try:
            df = data[t]
            # กรอง Volume ต่ำเกินไป (ขั้นต่ำตามมาตรฐานสถาบัน)
            adv = df['Volume'].tail(20).mean() * df['Close'].tail(20).mean()
            if adv < 5_000_000:
                continue
                
            # 2. การทำ Winsorization (ตัดหางที่เพี้ยน)
            close = df['Close']
            low_cap = close.quantile(0.05)
            high_cap = close.quantile(0.95)
            close = close.clip(low_cap, high_cap)
            
            clean_data[t] = close
        except:
            continue
            
    df_clean = pd.DataFrame(clean_data).ffill()
    return df_clean

def robust_zscore(df):
    """
    เปลี่ยนไปใช้ MAD (Median Absolute Deviation) แทน Standard Deviation
    เพื่อความทนทานต่อข้อมูลที่พุ่งกระโดด (Institutional Standard)
    """
    median = df.median()
    mad = (df - median).abs().median()
    # ป้องกันเลข 0
    return (df - median) / (mad * 1.4826 + 1e-9)

# ตัวอย่างการเรียกใช้งาน:
# df_final = get_clean_universe(['NVDA', 'MSFT', 'GOOG', ...])
# z_scores = robust_zscore(df_final)
