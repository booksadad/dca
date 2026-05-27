import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import StandardScaler
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError: pass 

class MarketRegimeHMM:
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42, init_params="smc")
        self.model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        
    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_macro_features(period="2y"): # ติดเทอร์โบจำกัดไว้ที่ 2y เพื่อตัดปัญหาโหลดดีเลย์ตอนเทสต์ระบบ
        tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "LQD"]
        data = yf.download(tickers, period=period, group_by='ticker', threads=True)
        df = pd.DataFrame()
        df['VIX'] = data['^VIX']['Close']
        df['SPY_Ret'] = data['SPY']['Close'].pct_change()
        df['Realized_Vol'] = df['SPY_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
        df['Vol_Premium'] = df['Realized_Vol'] - df['VIX']
        df['Yield_Curve'] = data['^TNX']['Close'] - data['^FVX']['Close']
        df['Credit_Stress'] = data['HYG']['Close'] / data['LQD']['Close']
        return df.dropna()

    @st.cache_data(ttl=86400)
    def expanding_fit_predict(_self, df_features, min_train_days=126):
        features = df_features[['VIX', 'Vol_Premium', 'Yield_Curve', 'Credit_Stress']].values
        scaler = StandardScaler()
        probs = np.zeros((len(features), _self.n_states))
        
        train_features = scaler.fit_transform(features[:min_train_days])
        _self.model.fit(train_features)
        probs[:min_train_days] = _self.model.predict_proba(train_features)
        
        for t in range(min_train_days, len(features)):
            scaled_train = scaler.fit_transform(features[:t]) 
            _self.model.fit(scaled_train)
            probs[t] = _self.model.predict_proba(scaler.transform(features[t:t+1]))[0]
            
        df_probs = pd.DataFrame(probs, index=df_features.index, columns=[f'State_{i}' for i in range(_self.n_states)])
        vix_means = [df_features.loc[df_probs[f'State_{i}'] > 0.5, 'VIX'].mean() for i in range(_self.n_states)]
        panic_idx = np.argmax(vix_means)
        return df_probs.rename(columns={f'State_{1-panic_idx}': 'P_BULL', f'State_{panic_idx}': 'P_PANIC'})

    def apply_transition_smoothing(self, df_probs, alpha=0.25):
        return df_probs.ewm(alpha=alpha, adjust=False).mean()

class DynamicFactorAllocator:
    def __init__(self):
        self.w_bull = {'Mom': 0.50, 'Qual': 0.30, 'Val': 0.20}
        self.w_panic = {'Mom': 0.10, 'Qual': 0.60, 'Val': 0.30}
        
    def calculate_weights(self, smoothed_probs):
        b, p = smoothed_probs['P_BULL'].iloc[-1], smoothed_probs['P_PANIC'].iloc[-1]
        return {'Mom': b*0.5+p*0.1, 'Qual': b*0.3+p*0.6, 'Val': b*0.2+p*0.3, 'P_BULL': b, 'P_PANIC': p}
