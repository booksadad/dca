import pandas as pd
import numpy as np
import yfinance as yf

class InstitutionalDataPipeline:
    def __init__(self, tickers):
        self.tickers = tickers

    def filter_universe(self, adv_threshold=5_000_000, mktcap_threshold=500_000_000):
        """
        [Layer 1.1] Universe Construction 
        ตัดหุ้น Market Cap เล็ก และสภาพคล่องต่ำ (ADV) ออกจากสารบบ
        """
        valid_tickers = []
        for t in self.tickers:
            try:
                stock = yf.Ticker(t)
                info = stock.info
                
                # เช็ค Market Cap Floor
                mkt_cap = info.get('marketCap', 0)
                if mkt_cap < mktcap_threshold:
                    continue
                
                # เช็ค Average Daily Volume (ADV) ย้อนหลัง 20 วัน
                hist = stock.history(period="1mo")
                if len(hist) > 0:
                    adv = (hist['Volume'] * hist['Close']).mean()
                    if adv >= adv_threshold:
                        valid_tickers.append(t)
            except:
                continue
        return valid_tickers

    def clean_fundamentals(self, df_fundamentals):
        """
        [Layer 1.2] Clean Fundamentals (#6)
        - TTM Averaging
        - 3-Sigma Outlier Hold (Flag)
        - PIT (Point-in-Time) Lagging
        """
        clean_df = df_fundamentals.copy()
        
        # สมมติว่า Columns เป็น Time-Series ของ ROA, FCF, Margin รายไตรมาส
        metrics = ['ROA', 'FCF', 'Margin']
        
        for metric in metrics:
            if metric in clean_df.columns:
                
                # 1. 3-Sigma Outlier Flag -> Hold ค่าเดิม (ป้องกัน One-off items)
                # คำนวณ Z-Score แบบ Rolling ย้อนหลัง
                rolling_mean = clean_df[metric].rolling(window=8, min_periods=1).mean()
                rolling_std = clean_df[metric].rolling(window=8, min_periods=1).std().fillna(method='bfill')
                
                z_score = (clean_df[metric] - rolling_mean) / (rolling_std + 1e-9)
                
                # สร้าง Mask สำหรับตัวที่กระโดดเกิน 3 Sigma
                outlier_mask = z_score.abs() > 3.0
                
                # ถ้าเจอ Outlier ให้แทนค่าด้วย NaN แล้วใช้ Forward Fill (Hold ค่าก่อนหน้า)
                clean_df.loc[outlier_mask, metric] = np.nan
                clean_df[metric] = clean_df[metric].ffill()

                # 2. ทำ TTM (เฉลี่ย 4 ไตรมาสย้อนหลัง ลด Spike)
                clean_df[f'{metric}_TTM'] = clean_df[metric].rolling(window=4, min_periods=1).mean()
                
                # 3. PIT (Point-in-Time) - Shift ข้อมูล 1 ไตรมาส (หรือ 45 วัน) 
                # ป้องกัน Look-ahead bias ว่าเรารู้งบการเงินตั้งแต่วันปิดงบ
                clean_df[f'{metric}_PIT'] = clean_df[f'{metric}_TTM'].shift(1).ffill()

        return clean_df
