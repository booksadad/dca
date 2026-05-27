import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

class InstitutionalDataPipeline:
    def __init__(self, tickers):
        self.tickers = tickers

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_bulk_market_data(tickers, period="2y"):
        data = yf.download(tickers, period=period, group_by='ticker', threads=True)
        return data

    def filter_universe(self, adv_threshold=5_000_000, price_floor=5.0, min_history_days=252):
        valid_tickers = []
        raw_data = self.fetch_bulk_market_data(self.tickers)
        
        for t in self.tickers:
            try:
                df = raw_data[t] if len(self.tickers) > 1 else raw_data
                df = df.dropna(subset=['Close'])
                if len(df) < min_history_days or df['Close'].iloc[-1] < price_floor:
                    continue
                adv = (df['Volume'].tail(20) * df['Close'].tail(20)).mean()
                if adv >= adv_threshold:
                    valid_tickers.append(t)
            except: continue
        return valid_tickers

    def clean_fundamentals(self, df_fundamentals):
        clean_df = df_fundamentals.copy()
        metrics = ['ROA', 'FCF', 'Margin']
        for metric in metrics:
            if metric in clean_df.columns:
                rolling_median = clean_df[metric].rolling(window=8, min_periods=1).median()
                rolling_mad = clean_df[metric].rolling(window=8, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
                mad_scaled = rolling_mad * 1.4826 + 1e-9
                clean_df[metric] = clean_df[metric].clip(lower=rolling_median - 3.0 * mad_scaled, upper=rolling_median + 3.0 * mad_scaled)
                clean_df[f'{metric}_TTM'] = clean_df[metric].rolling(window=4, min_periods=1).mean()
                clean_df[f'{metric}_PIT'] = clean_df[f'{metric}_TTM'].shift(1).ffill()
        return clean_df
