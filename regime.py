import pandas as pd
import numpy as np
import yfinance as yf
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    pass # ดักไว้ก่อน เผื่อยังไม่ได้ลง hmmlearn

class MarketRegimeHMM:
    def __init__(self, n_states=2):
        # เริ่มด้วย 2 States (BULL / PANIC) ตามที่เพื่อนแนะนำ เพื่อความเสถียรก่อน
        self.n_states = n_states
        self.model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100, random_state=42)
        
    def fetch_macro_features(self, period="5y"):
        """
        [Layer 3.1] Regime Input Features
        ดึงข้อมูล Macro: VIX, Realized Volatility และ Yield Curve Proxy
        """
        # ดึง SPY (แทนตลาด), VIX (Fear Gauge), TNX (10Y Yield), FVX (5Y Yield แทน 2Y ขัดตาทัพใน Yahoo)
        tickers = ["SPY", "^VIX", "^TNX", "^FVX"]
        data = yf.download(tickers, period=period, group_by='ticker')
        
        df = pd.DataFrame()
        # 1. VIX Level
        df['VIX'] = data['^VIX']['Close']
        
        # 2. Realized Vol (20d) เทียบกับ Implied Vol (VIX)
        df['SPY_Ret'] = data['SPY']['Close'].pct_change()
        df['Realized_Vol'] = df['SPY_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
        df['Vol_Premium'] = df['Realized_Vol'] - df['VIX'] # ถ้า Realized > Implied = Panic เริ่มแล้ว
        
        # 3. Yield Curve Proxy (10Y - 5Y)
        df['Yield_Curve'] = data['^TNX']['Close'] - data['^FVX']['Close']
        
        return df.dropna()

    def fit_and_predict_proba(self, df_features):
        """
        [Layer 3.2] HMM Setup
        Fit โมเดลแบบ Unsupervised เพื่อหาความน่าจะเป็นของแต่ละ Regime
        """
        # เลือก Features ที่จะใช้เทรน
        features = df_features[['VIX', 'Vol_Premium', 'Yield_Curve']].values
        
        # 1. Fit Model (ควรทำเป็น Expanding Window ใน Production จริง)
        self.model.fit(features)
        
        # 2. Predict Probabilities (P(State 0), P(State 1))
        probs = self.model.predict_proba(features)
        df_probs = pd.DataFrame(probs, index=df_features.index, columns=[f'State_{i}' for i in range(self.n_states)])
        
        # จัดพจน์ให้ State ที่ VIX สูงกว่า = PANIC
        vix_means = [df_features.loc[df_probs[f'State_{i}'] > 0.5, 'VIX'].mean() for i in range(self.n_states)]
        panic_state_idx = np.argmax(vix_means)
        bull_state_idx = 1 - panic_state_idx
        
        df_probs = df_probs.rename(columns={
            f'State_{bull_state_idx}': 'P_BULL',
            f'State_{panic_state_idx}': 'P_PANIC'
        })
        
        return df_probs

    def apply_transition_smoothing(self, df_probs, alpha=0.25):
        """
        [Layer 3.3] Soft Weighting + Transition Smoothing
        EMA Smooth ป้องกัน Weights กระโดดจาก Noise รายวัน
        สมการ: P_smooth(t) = 0.25 * P_HMM(t) + 0.75 * P_smooth(t-1)
        """
        # ใช้ ewm (Exponential Weighted Math) ของ pandas จัดการให้เลย
        df_smoothed = df_probs.ewm(alpha=alpha, adjust=False).mean()
        return df_smoothed

    def calculate_dynamic_weights(self, smoothed_probs):
        """
        แปลง Probability เป็น Factor Weights
        Blend น้ำหนักแบบไร้รอยต่อ ไม่มี Hard Switch
        """
        # ตั้งค่าน้ำหนักพื้นฐาน (Base Weights)
        # BULL: เร่ง Momentum / PANIC: หนีเข้า Quality
        w_bull = {'Mom': 0.50, 'Qual': 0.30, 'Val': 0.20}
        w_panic = {'Mom': 0.10, 'Qual': 0.60, 'Val': 0.30}
        
        # เอา Probability แถวสุดท้ายมาใช้ (ของวันล่าสุด)
        latest_p_bull = smoothed_probs['P_BULL'].iloc[-1]
        latest_p_panic = smoothed_probs['P_PANIC'].iloc[-1]
        
        # Blend Weights (Dot Product)
        final_w_mom = (latest_p_bull * w_bull['Mom']) + (latest_p_panic * w_panic['Mom'])
        final_w_qual = (latest_p_bull * w_bull['Qual']) + (latest_p_panic * w_panic['Qual'])
        final_w_val = (latest_p_bull * w_bull['Val']) + (latest_p_panic * w_panic['Val'])
        
        return {
            'Mom': final_w_mom,
            'Qual': final_w_qual,
            'Val': final_w_val,
            'P_BULL': latest_p_bull,
            'P_PANIC': latest_p_panic
        }

# ==========================================
# วิธีเรียกใช้งานในไฟล์หลัก (1_📊_Smart_DCA.py)
# ==========================================
# hmm_engine = MarketRegimeHMM(n_states=2)
# df_macro = hmm_engine.fetch_macro_features()
# raw_probs = hmm_engine.fit_and_predict_proba(df_macro)
# smooth_probs = hmm_engine.apply_transition_smoothing(raw_probs, alpha=0.25)
# current_regime_weights = hmm_engine.calculate_dynamic_weights(smooth_probs)
#
# ตอนนี้ current_regime_weights['Mom'] จะค่อยๆ สไลด์ขึ้นลงแบบเนียนตา ไม่กระโดดไปมา!
