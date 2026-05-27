import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

class InstitutionalDataPipeline:
    def __init__(self, tickers):
        self.tickers = tickers

    @staticmethod
    @st.cache_data(ttl=86400) # 💾 Cache Layer: เก็บข้อมูลไว้ 1 วัน (24 ชม.)
    def fetch_bulk_market_data(tickers, period="2y"):
        """
        โหลดข้อมูลแบบ Bulk รวดเดียวจบ เพื่อแก้ปัญหา yf.info ที่ช้าและชอบ Timeout
        """
        data = yf.download(tickers, period=period, group_by='ticker', threads=True)
        return data

    def filter_universe(self, adv_threshold=5_000_000, price_floor=5.0, min_history_days=252):
        """
        [Layer 1.1] Universe Construction (Robust Filter)
        - ADV > $5M
        - Price > $5 (ตัด Penny Stocks)
        - Listing Age > 1 ปี (ตัด IPO Noise)
        """
        valid_tickers = []
        # ใช้ Cache Layer โหลดข้อมูลราคาย้อนหลัง 2 ปี
        raw_data = self.fetch_bulk_market_data(self.tickers)
        
        for t in self.tickers:
            try:
                # จัดการกรณีโหลดหุ้นตัวเดียว vs หลายตัว
                df = raw_data[t] if len(self.tickers) > 1 else raw_data
                df = df.dropna(subset=['Close'])
                
                # 1. Listing Age Filter (ต้องมีข้อมูลเทรดมาแล้วอย่างน้อย ~1 ปี)
                if len(df) < min_history_days:
                    continue
                    
                # 2. Price Floor (ราคาปิดล่าสุดต้องมากกว่า $5)
                if df['Close'].iloc[-1] < price_floor:
                    continue
                    
                # 3. ADV Filter (Volume * Price ย้อนหลัง 20 วัน)
                adv = (df['Volume'].tail(20) * df['Close'].tail(20)).mean()
                if adv >= adv_threshold:
                    valid_tickers.append(t)
            except:
                continue
                
        return valid_tickers

    def clean_fundamentals(self, df_fundamentals):
        """
        [Layer 1.2] Clean Fundamentals (Advanced Quant Methods)
        - Rolling MAD แทน Standard Deviation
        - Soft Clipping แทน Forward Fill
        - PIT (Point-in-Time) Lagging
        """
        clean_df = df_fundamentals.copy()
        metrics = ['ROA', 'FCF', 'Margin']
        
        for metric in metrics:
            if metric in clean_df.columns:
                
                # 🧮 1. Robust Outlier Detection (Rolling Median + Rolling MAD)
                rolling_median = clean_df[metric].rolling(window=8, min_periods=1).median()
                
                # คำนวณ MAD: Median of the Absolute Deviations
                rolling_mad = clean_df[metric].rolling(window=8, min_periods=1).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                
                # คำนวณขอบเขตบนและล่าง (±3 Robust Z-Score)
                mad_scaled = rolling_mad * 1.4826 + 1e-9
                upper_bound = rolling_median + (3.0 * mad_scaled)
                lower_bound = rolling_median - (3.0 * mad_scaled)
                
                # ✂️ 2. Soft Clipping 
                # จำกัดเพดานและพื้นข้อมูล ไม่ให้เกินขอบเขต 3 MAD (แต่ไม่ใช้ ffill แช่แข็ง)
                clean_df[metric] = clean_df[metric].clip(lower=lower_bound, upper=upper_bound)

                # 3. TTM Smoothing (Prototype Approach)
                # หมายเหตุ: ใน production จริงควรใช้ (Net Income TTM / Avg Assets)
                # แต่เนื่องจากข้อจำกัดของ yfinance เราใช้ Rolling Mean 4 ไตรมาสไปก่อน
                clean_df[f'{metric}_TTM'] = clean_df[metric].rolling(window=4, min_periods=1).mean()
                
                # 4. PIT Lagging (Shift 1 Qtr)
                clean_df[f'{metric}_PIT'] = clean_df[f'{metric}_TTM'].shift(1).ffill()

        return clean_df
