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
    def fetch_macro_features(period="2y"): 
        tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "LQD"]
        data = yf.download(tickers, period=period, threads=True, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            close_data = data
            
        df = pd.DataFrame()
        
        # ใช้ ffill() อุดรอยรั่ววันหยุดตลาดพันธบัตร
        df['VIX'] = close_data['^VIX'].ffill()
        df['SPY_Ret'] = close_data['SPY'].ffill().pct_change()
        df['Realized_Vol'] = df['SPY_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
        df['Vol_Premium'] = df['Realized_Vol'] - df['VIX']
        df['Yield_Curve'] = close_data['^TNX'].ffill() - close_data['^FVX'].ffill()
        df['Credit_Stress'] = close_data['HYG'].ffill() / close_data['LQD'].ffill()
        
        return df.dropna()

    @st.cache_data(ttl=86400)
    def expanding_fit_predict(_self, df_features, min_train_days=126):
        if len(df_features) < min_train_days + 10:
            fallback = pd.DataFrame(index=df_features.index)
            fallback['P_BULL'] = 0.5
            fallback['P_PANIC'] = 0.5
            return fallback

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

    # ==========================================
    # 🌊 PRIORITY 3: HYSTERESIS BAND (ลด Whipsaw)
    # ==========================================
    def apply_hysteresis(self, df_probs, upper_threshold=0.70, lower_threshold=0.30):
        states = []
        current_state = "BULL" # สมมติจุดเริ่มต้น
        
        for p_panic in df_probs['P_PANIC']:
            if current_state == "BULL":
                if p_panic >= upper_threshold: current_state = "PANIC"
            elif current_state == "PANIC":
                if p_panic <= lower_threshold: current_state = "BULL"
            states.append(current_state)
            
        df_probs['HMM_State'] = states
        df_probs['Adj_P_PANIC'] = [1.0 if s == "PANIC" else 0.0 for s in states]
        df_probs['Adj_P_BULL'] = [1.0 if s == "BULL" else 0.0 for s in states]
        
        return df_probs

class DynamicFactorAllocator:
    def __init__(self):
        self.w_bull = {'Mom': 0.50, 'Qual': 0.30, 'Val': 0.20}
        self.w_panic = {'Mom': 0.10, 'Qual': 0.60, 'Val': 0.30}
        
    def calculate_weights(self, df_probs_with_hysteresis):
        # ใช้ความน่าจะเป็นที่ผ่านความเฉื่อยแล้วมาถ่วง Factor
        b = df_probs_with_hysteresis['Adj_P_BULL'].iloc[-1]
        p = df_probs_with_hysteresis['Adj_P_PANIC'].iloc[-1]
        
        return {
            'Mom': (b * self.w_bull['Mom']) + (p * self.w_panic['Mom']), 
            'Qual': (b * self.w_bull['Qual']) + (p * self.w_panic['Qual']), 
            'Val': (b * self.w_bull['Val']) + (p * self.w_panic['Val']), 
            'P_BULL': b, 
            'P_PANIC': p,
            'Current_State': df_probs_with_hysteresis['HMM_State'].iloc[-1]
        }
