import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    pass 

# ==========================================
# 1. ระบบตรวจจับสภาวะตลาด (Regime Detection)
# ==========================================
class MarketRegimeHMM:
    def __init__(self, n_states=2):
        self.n_states = n_states
        
        # 🛡️ Priority 3: Regime Persistence Constraint
        # init_params="stmc" แปลว่าให้โมเดลเรียนรู้ทุกอย่าง "ยกเว้น" Transition Matrix (t)
        self.model = GaussianHMM(
            n_components=self.n_states, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42,
            init_params="smc" # เราจะตั้งค่า transmat_ (t) เอง
        )
        
        # บังคับให้มีความเฉื่อย (Sticky Regime) 95% โอกาสที่พรุ่งนี้จะเป็น Regime เดิม
        self.model.transmat_ = np.array([[0.95, 0.05], 
                                         [0.05, 0.95]])
        
    def fetch_macro_features(self, period="5y"):
        """ดึง Features ที่กระจายความเสี่ยง (Orthogonal) ตามสถาบันใช้"""
        # เพิ่ม HYG (Junk Bonds) และ LQD (Investment Grade) เพื่อดู Credit Stress
        tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "LQD"]
        data = yf.download(tickers, period=period, group_by='ticker', threads=True)
        
        df = pd.DataFrame()
        df['VIX'] = data['^VIX']['Close']
        
        df['SPY_Ret'] = data['SPY']['Close'].pct_change()
        df['Realized_Vol'] = df['SPY_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
        df['Vol_Premium'] = df['Realized_Vol'] - df['VIX']
        
        df['Yield_Curve'] = data['^TNX']['Close'] - data['^FVX']['Close']
        
        # 🛡️ New Feature: Credit Spread (HYG/LQD)
        # ถ้านักลงทุนกลัว จะขาย HYG ไปซื้อ LQD ทำให้ Ratio นี้ดิ่งลง
        df['Credit_Stress'] = data['HYG']['Close'] / data['LQD']['Close']
        
        return df.dropna()

    def expanding_fit_predict(self, df_features, min_train_days=252):
        """
        🛡️ Priority 2: Expanding-window training
        ป้องกัน Hindsight bias โดยสอนโมเดลถึงแค่เมื่อวาน แล้วทำนายวันนี้
        """
        features = df_features[['VIX', 'Vol_Premium', 'Yield_Curve', 'Credit_Stress']].values
        
        # 🛡️ Priority 1: Standardize features
        scaler = StandardScaler()
        
        probs = np.zeros((len(features), self.n_states))
        
        # Initial Fit (ใช้เวลา 1 ปีแรกในการตั้งไข่โมเดล)
        train_features = scaler.fit_transform(features[:min_train_days])
        self.model.fit(train_features)
        probs[:min_train_days] = self.model.predict_proba(train_features)
        
        # Expanding Window Loop (เดินหน้าทีละวัน)
        for t in range(min_train_days, len(features)):
            # ดึงข้อมูลตั้งแต่วันแรกจนถึง "เมื่อวาน" (t-1)
            train_slice = features[:t]
            scaled_train = scaler.fit_transform(train_slice) 
            self.model.fit(scaled_train)
            
            # ดึงข้อมูล "วันนี้" (t) มา Predict
            test_slice = features[t:t+1]
            scaled_test = scaler.transform(test_slice) # ใช้ scaler ของอดีต
            probs[t] = self.model.predict_proba(scaled_test)[0]
            
        df_probs = pd.DataFrame(probs, index=df_features.index, columns=[f'State_{i}' for i in range(self.n_states)])
        
        # แปะป้ายว่า State ไหนคือ Panic (ดูที่ State ไหนค่าเฉลี่ย VIX สูงกว่า)
        vix_means = [df_features.loc[df_probs[f'State_{i}'] > 0.5, 'VIX'].mean() for i in range(self.n_states)]
        panic_state_idx = np.argmax(vix_means)
        bull_state_idx = 1 - panic_state_idx
        
        df_probs = df_probs.rename(columns={
            f'State_{bull_state_idx}': 'P_BULL',
            f'State_{panic_state_idx}': 'P_PANIC'
        })
        
        return df_probs

    def apply_transition_smoothing(self, df_probs, alpha=0.25):
        """EMA Smooth เพื่อลด Whipsaw"""
        return df_probs.ewm(alpha=alpha, adjust=False).mean()

# ==========================================
# 2. ระบบจัดสรรน้ำหนัก (Factor Allocator)
# ==========================================
class DynamicFactorAllocator:
    """
    🛡️ Priority 4: Separate regime detection กับ factor weighting (Modularity)
    """
    def __init__(self):
        # โครงสร้างนี้สามารถขยายเป็น 5 Regimes ได้ในอนาคตตามที่เพื่อนแนะนำ
        self.w_bull = {'Mom': 0.50, 'Qual': 0.30, 'Val': 0.20}
        self.w_panic = {'Mom': 0.10, 'Qual': 0.60, 'Val': 0.30}
        
    def calculate_weights(self, smoothed_probs):
        latest_p_bull = smoothed_probs['P_BULL'].iloc[-1]
        latest_p_panic = smoothed_probs['P_PANIC'].iloc[-1]
        
        final_w_mom = (latest_p_bull * self.w_bull['Mom']) + (latest_p_panic * self.w_panic['Mom'])
        final_w_qual = (latest_p_bull * self.w_bull['Qual']) + (latest_p_panic * self.w_panic['Qual'])
        final_w_val = (latest_p_bull * self.w_bull['Val']) + (latest_p_panic * self.w_panic['Val'])
        
        return {
            'Mom': final_w_mom,
            'Qual': final_w_qual,
            'Val': final_w_val,
            'P_BULL': latest_p_bull,
            'P_PANIC': latest_p_panic
        }
