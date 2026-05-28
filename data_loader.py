import yfinance as yf
import pandas as pd
import numpy as np
from cache_utils import ttl_cache
import os, sys

def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

@ttl_cache(ttl_seconds=3600)
def fetch_fundamental_data(tickers):
    metrics = []
    for t in tickers:
        try:
            block_print()
            info = yf.Ticker(t).info
            enable_print()
            roa = info.get('returnOnAssets', 0.0)
            margin = info.get('profitMargins', 0.0)
            peg = info.get('pegRatio', 1.0)
            fcf = info.get('freeCashflow', 0.0)
            mcap = info.get('marketCap', 1.0)
            fcf_yield = (fcf / mcap) if pd.notna(fcf) and pd.notna(mcap) and mcap > 0 else 0.0
        except:
            enable_print()
            roa, margin, peg, fcf_yield = 0.0, 0.0, 1.0, 0.0
            
        metrics.append({
            'Ticker': t, 'ROA': roa * 100 if pd.notna(roa) else 0.0,
            'Margin': margin * 100 if pd.notna(margin) else 0.0,
            'PEG': peg if pd.notna(peg) else 1.0, 
            'FCF_Yield': fcf_yield * 100 if pd.notna(fcf_yield) else 0.0
        })
    return pd.DataFrame(metrics)

@ttl_cache(ttl_seconds=1800)
def fetch_market_data(tickers, period="3y"):
    block_print()
    data = yf.download(tickers, period=period, progress=False)['Close']
    enable_print()
    return data
