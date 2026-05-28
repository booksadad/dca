import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

class InstitutionalDataPipeline:
    def __init__(self, universe):
        self.universe = universe

    # ==========================================
    # 🚨 PRIORITY 1: SANITY CHECK LAYER 
    # ==========================================
    def validate_price_data(self, df):
        """
        ดักจับและทำความสะอาดข้อมูลขยะจาก Yahoo Finance ก่อนส่งเข้าโมเดล
        """
        clean_df = df.copy()
        
        # 1. Remove impossible jumps (ดัก Gap ผิดปกติเกิน 20% ที่มักเกิดจากบั๊กแตกพาร์)
        returns = clean_df['Close'].pct_change()
        bad_gap = returns.abs() > 0.20
        clean_df.loc[bad_gap, 'Close'] = np.nan
        
        # 2. Forward fill temporary glitches (อุดรอยรั่วจาก Gap ที่โดนลบ และวันหยุดตลาด)
        clean_df['Close'] = clean_df['Close'].ffill()
        
        # 3. Remove negative / zero prices (ดักบั๊กราคาติดลบหรือเป็นศูนย์)
        clean_df = clean_df[clean_df['Close'] > 0]
        
        return clean_df

    @st.cache_data(ttl=3600)
    def fetch_bulk_market_data(_self, tickers, period="2y"):
        """
        ดึงข้อมูลราคาและส่งเข้า Sanity Check ทันที
        """
        data = yf.download(tickers, period=period, threads=True, progress=False)
        
        # แยกเฉพาะราคา Close
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            close_data = data
            
        validated_dict = {}
        for t in tickers:
            try:
                # ดึงราคาหุ้นทีละตัวมาทำ Sanity Check
                temp_df = pd.DataFrame({'Close': close_data[t]})
                clean_temp = _self.validate_price_data(temp_df)
                validated_dict[t] = clean_temp['Close']
            except:
                pass
                
        # รวมร่างกลับเป็น DataFrame ที่สะอาดหมดจด
        return pd.DataFrame(validated_dict)

    def filter_universe(self):
        # โค้ดกรองหุ้นขยะ (ADV / Price Floor) ของเดิม...
        return self.universe
