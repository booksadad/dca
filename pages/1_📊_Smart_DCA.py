import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os, sys
import json
import plotly.express as px
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from sklearn.covariance import LedoitWolf 

warnings.filterwarnings("ignore")

# ==========================================
# 🛠️ ฟังก์ชันคณิตศาสตร์และเครื่องมือ Quant
# ==========================================
def calc_zscore(series): 
    if series.std() == 0: return series - series.mean()
    return (series - series.mean()) / series.std()

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_doi_risk(rsi_val):
    if rsi_val > 75: return 'ดอย (ซื้อระวัง)'
    elif rsi_val < 30: return 'ของถูก (เก็บสะสม)'
    return 'ปกติ'

def find_sr_levels(series):
    try:
        current_price = series.iloc[-1]
        local_mins = series.iloc[argrelextrema(series.values, np.less_equal, order=5)[0]]
        local_maxs = series.iloc[argrelextrema(series.values, np.greater_equal, order=5)[0]]
        supports = local_mins[local_mins < current_price].sort_values(ascending=False).unique()
        resistances = local_maxs[local_maxs > current_price].sort_values(ascending=True).unique()
        s1 = supports[0] if len(supports) > 0 else current_price * 0.95
        s2 = supports[1] if len(supports) > 1 else s1 * 0.95
        r1 = resistances[0] if len(resistances) > 0 else current_price * 1.05
        return f"รับ: {s1:.1f}, {s2:.1f} | ต้าน: {r1:.1f}"
    except: return "-"

def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

@st.cache_data(ttl=3600) # 🌟 Caching yfinance ป้องกันอืดและโดนแบน
def fetch_fundamental_data(tickers):
    metrics = []
    for t in tickers:
        try:
            block_print()
            info = yf.Ticker(t).info
            enable_print()
            roa = info.get('returnOnAssets', np.nan)
            margin = info.get('profitMargins', np.nan)
            peg = info.get('pegRatio', np.nan)
            fcf = info.get('freeCashflow', np.nan)
            mcap = info.get('marketCap', np.nan)
            fcf_yield = (fcf / mcap) if pd.notna(fcf) and pd.notna(mcap) and mcap > 0 else np.nan
        except:
            enable_print()
            roa, margin, peg, fcf_yield = np.nan, np.nan, np.nan, np.nan
            
        metrics.append({
            'Ticker': t, 'ROA': roa * 100 if pd.notna(roa) else np.nan,
            'Margin': margin * 100 if pd.notna(margin) else np.nan,
            'PEG': peg, 'FCF_Yield': fcf_yield * 100 if pd.notna(fcf_yield) else np.nan
        })
    return pd.DataFrame(metrics)

# ==========================================
# 💾 ฐานข้อมูลถอดรหัสและ Thesis Layer 
# ==========================================
PORTFOLIO_FILE = "my_portfolio_data.csv"

SECTOR_DB = {
    "💻 Tech": ["NVDA", "MSFT", "GOOG", "META", "CSCO", "TXN", "AAPL", "AMD", "PLTR", "AVGO", "RKLB"],
    "🛍️ Consumer": ["COST", "KO", "PEP", "BLK", "MELI", "V", "MA", "WMT", "JPM"],
    "🩺 Health": ["UNH", "JNJ", "ISRG", "LLY", "ABBV"],
    "🚗 EV": ["TSLA", "UBER", "TM"],
    "🌿 Green": ["FSLR", "ENPH", "NEE", "SEDG"]
}
ticker_to_sector = {ticker: sector for sector, tickers in SECTOR_DB.items() for ticker in tickers}

THESIS_DB = {
    "NVDA": "AI Infra Dominance", "MSFT": "Cloud & OS Monopoly",
    "COST": "Membership Cashflow", "AVGO": "Custom Silicon / M&A",
    "V": "Global Toll Network", "KO": "Defensive Cashflow",
    "JNJ": "Healthcare Titan", "RKLB": "Space Infrastructure"
}

if 'dca_budget' not in st.session_state: st.session_state['dca_budget'] = 500.0 
if 'min_order_thb' not in st.session_state: st.session_state['min_order_thb'] = 50.0

st.set_page_config(page_title="QuantHQ DCA", page_icon="🛡️", layout="wide")
st.title("🛡️ QUANT-HQ DCA (V.Institutional Awakening)")
st.markdown("ระบบจัดพอร์ตระดับสถาบัน **(Ledoit-Wolf, Dynamic Factors, Optimizer Constraints)**")
st.markdown("---")

# ==========================================
# 🗂️ แถบด้านซ้าย (Sidebar)
# ==========================================
default_tickers = "MSFT, AVGO, COST, NVDA, V, KO, JNJ, RKLB"
if os.path.exists(PORTFOLIO_FILE):
    try:
        temp_df = pd.read_csv(PORTFOLIO_FILE)
        if not temp_df.empty and "รายชื่อหุ้น" in temp_df.columns:
            valid_tickers = [str(t) for t in temp_df["รายชื่อหุ้น"].tolist() if str(t).strip()]
            if valid_tickers: default_tickers = ", ".join(valid_tickers)
    except: pass

st.sidebar.subheader("🗂️ หุ้นในพอร์ตของคุณ")
tickers_input = st.sidebar.text_area("รายชื่อหุ้นที่ถืออยู่ (คั่นด้วยลูกน้ำ)", default_tickers)
my_portfolio = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.sidebar.markdown("---")
engine_choice = st.sidebar.radio("เข็มทิศการลงทุน:", ["🧠 Auto-Pilot (BL + Adaptive)", "🛡️ Safe Mode (Risk Parity)"])
base_lambda = st.sidebar.slider("ระดับความกลัวตั้งต้น", 0.1, 10.0, 2.0, 0.1)

col_input1, col_input2 = st.columns(2)
with col_input1: 
    budget = st.number_input("💵 งบประมาณจัดสรร (บาท)", 100, 50000, int(st.session_state['dca_budget']), 100)
    st.session_state['dca_budget'] = float(budget)
with col_input2: 
    min_order_thb = st.number_input("🚦 ยอดซื้อขั้นต่ำต่อหุ้น (บาท)", 10, 500, int(st.session_state['min_order_thb']), 10)
    st.session_state['min_order_thb'] = float(min_order_thb)

st.markdown("### 💼 1. ตรวจสอบพอร์ตปัจจุบัน")

saved_dict = {}
if os.path.exists(PORTFOLIO_FILE):
    try:
        saved_df = pd.read_csv(PORTFOLIO_FILE)
        if not saved_df.empty and "รายชื่อหุ้น" in saved_df.columns:
            saved_dict = dict(zip(saved_df["รายชื่อหุ้น"], pd.to_numeric(saved_df["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0.0)))
    except: pass

default_rows = [{"รายชื่อหุ้น": t, "ยอดเงินปัจจุบัน (บาท)": float(saved_dict.get(t, 0.0))} for t in my_portfolio]

if 'portfolio_holdings' not in st.session_state:
    st.session_state['portfolio_holdings'] = pd.DataFrame(default_rows)
else:
    current_tickers_in_state = st.session_state['portfolio_holdings']['รายชื่อหุ้น'].tolist()
    if current_tickers_in_state != my_portfolio:
        st.session_state['portfolio_holdings'] = pd.DataFrame(default_rows)

df_holdings_edited = st.data_editor(st.session_state['portfolio_holdings'], use_container_width=True, hide_index=True)
df_holdings_edited["ยอดเงินปัจจุบัน (บาท)"] = pd.to_numeric(df_holdings_edited["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0.0)
st.session_state['portfolio_holdings'] = df_holdings_edited
df_holdings_edited.to_csv(PORTFOLIO_FILE, index=False)
current_thb = dict(zip(df_holdings_edited["รายชื่อหุ้น"], df_holdings_edited["ยอดเงินปัจจุบัน (บาท)"]))

# 🛠️ ซ่อมบั๊กงบประมาณหาย
actual_budget = budget 

# 🛠️ ซ่อมบั๊กปุ่มเด้ง
if 'run_quant_engine' not in st.session_state: st.session_state['run_quant_engine'] = False
if st.button("🚀 รันระบบ Quant Matrix", type="primary"): st.session_state['run_quant_engine'] = True

if st.session_state['run_quant_engine']:
    if not my_portfolio: 
        st.error("⚠️ โปรดระบุชื่อหุ้นก่อนครับ")
        st.session_state['run_quant_engine'] = False
    else:
        status_box = st.status(f"🔮 เดินเครื่องสมองกลประมวลผล Matrix สากล...", expanded=True)
        
        benchmark = 'VOO'
        vix_ticker = '^VIX'
        
        block_print()
        market_data = yf.download([benchmark, vix_ticker], period="3y", progress=False)['Close']
        enable_print()
        
        vix_current = market_data[vix_ticker].iloc[-1] if vix_ticker in market_data else 20.0
        
        # 🌟 Portfolio Regime Engine 
        is_panic = False
        turnover_penalty = 0.02 
        max_sector_cap = 0.40 
        
        if vix_current > 25:
            st.error(f"🚨 **Regime: PANIC (VIX = {vix_current:.1f})** ลด Momentum, เพิ่มน้ำหนัก Quality, บังคับ Sector < 30%")
            is_panic = True
            turnover_penalty = 0.05
            w_mom, w_quality, w_value = 0.2, 0.5, 0.3 
            max_sector_cap = 0.30
        elif vix_current < 15:
            st.success(f"🐂 **Regime: BULL (VIX = {vix_current:.1f})** เร่งเครื่อง Momentum, ขยาย Sector Limit เป็น 50%")
            turnover_penalty = 0.01
            w_mom, w_quality, w_value = 0.5, 0.3, 0.2 
            max_sector_cap = 0.50
        else:
            st.info(f"⚖️ **Regime: NORMAL (VIX = {vix_current:.1f})** ตลาดอยู่ในสภาวะปกติ")
            w_mom, w_quality, w_value = 0.4, 0.3, 0.3
            max_sector_cap = 0.40

        voo_ret = market_data[benchmark].pct_change().dropna()
        vol_30d = voo_ret.tail(30).std() * np.sqrt(252)
        vol_252d = voo_ret.tail(252).std() * np.sqrt(252)
        vol_ratio = vol_30d / vol_252d if vol_252d > 0 else 1.0
        dynamic_lambda = base_lambda * vol_ratio
        
        market_proxy = [t for sublist in SECTOR_DB.values() for t in sublist]
        universe_list = list(set(my_portfolio + market_proxy)) 
        
        block_print()
        prices_3y = yf.download(universe_list, period="3y", progress=False)['Close']
        enable_print()
        
        prices_1y = prices_3y.tail(252) 
        returns_1y = prices_1y.pct_change().dropna()
        
        roll_max = prices_1y.cummax()
        drawdown = (prices_1y - roll_max) / roll_max
        max_dd = drawdown.min() * 100
        
        # 🧬 The Real Alpha (Residual Momentum & Cached Fundamentals)
        df_fundamentals = fetch_fundamental_data(prices_1y.columns.tolist())
        metrics = []
        rsi_data, sr_data = {}, {}
        
        for t in prices_1y.columns:
            try:
                cov = np.cov(returns_1y[t], voo_ret)[0][1]
                var = np.var(voo_ret)
                beta = cov / var if var > 0 else 1.0
                
                ret_6m = prices_1y[t].pct_change(periods=126).iloc[-1]
                mkt_ret_6m = market_data[benchmark].pct_change(periods=126).iloc[-1]
                residual_mom = ret_6m - (beta * mkt_ret_6m) 
            except: 
                residual_mom = np.nan
                
            metrics.append({'Ticker': t, 'Residual_Mom': residual_mom})
            
            series_clean = prices_1y[t].dropna()
            rsi_data[t] = calculate_rsi(series_clean).iloc[-1] if len(series_clean) > 14 else 50.0
            sr_data[t] = find_sr_levels(series_clean)
            
        df_metrics = pd.merge(pd.DataFrame(metrics), df_fundamentals, on='Ticker')
        for col in ['Residual_Mom', 'ROA', 'Margin', 'PEG', 'FCF_Yield']:
            df_metrics[col] = df_metrics[col].fillna(df_metrics[col].median())
            
        # คำนวณ Alpha Score แบบ Dynamic Factor Weighting
        z_mom = calc_zscore(df_metrics['Residual_Mom'])
        z_quality = (calc_zscore(df_metrics['ROA']) + calc_zscore(df_metrics['Margin'])) / 2
        z_value = (calc_zscore(df_metrics['FCF_Yield']) + (calc_zscore(df_metrics['PEG']) * -1)) / 2
        
        df_metrics['Alpha_Score'] = (z_mom * w_mom) + (z_quality * w_quality) + (z_value * w_value)
        df_metrics['Max_Drawdown'] = df_metrics['Ticker'].map(max_dd)
        final_df = df_metrics.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        port_df['Sector'] = port_df['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0

        # ==========================================
        # ⚖️ THE QUANT ENGINE (PYTHON LAYER)
        # ==========================================
        if "Auto-Pilot" in engine_choice:
            status_box.update(label=f"🧮 กำลังคำนวณ Dynamic Black-Litterman ด้วย Ledoit-Wolf Shrinkage...")
            port_tickers = port_df['Ticker'].tolist()
            port_returns = returns_1y[port_tickers]
            num_assets = len(port_tickers)
            
            try:
                # 🌟 Ledoit-Wolf Covariance Shrinkage 
                lw = LedoitWolf()
                lw.fit(port_returns)
                shrunk_cov = lw.covariance_ * 252 
                
                inv_vol = 1.0 / returns_1y[port_tickers].std()
                w_eq = (inv_vol / inv_vol.sum()).values
                Pi = dynamic_lambda * np.dot(shrunk_cov, w_eq) 
                
                tau = 0.05
                Q = np.zeros(num_assets)
                omega_diag = np.zeros(num_assets)
                P = np.eye(num_assets) 
                
                # 🧮 Dynamic Views & Sigmoid Conviction
                for i, t in enumerate(port_tickers):
                    alpha_score = port_df.loc[port_df['Ticker'] == t, 'Alpha_Score'].values[0]
                    
                    conviction = sigmoid(alpha_score) 
                    signal_strength = (conviction * 0.25) - 0.10 
                    Q[i] = Pi[i] + signal_strength
                    
                    base_uncertainty = shrunk_cov[i, i] * tau
                    omega_diag[i] = base_uncertainty / max(conviction, 0.01)

                Omega = np.diag(omega_diag)
                tau_cov_inv = np.linalg.inv(tau * shrunk_cov)
                omega_inv = np.linalg.inv(Omega)
                
                term1 = np.linalg.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
                term2 = np.dot(tau_cov_inv, Pi) + np.dot(np.dot(P.T, omega_inv), Q)
                mu_bl = np.dot(term1, term2) 

                current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(num_assets)
                
                def neg_utility(w):
                    expected_return = np.sum(mu_bl * w)
                    portfolio_variance = np.dot(w.T, np.dot(shrunk_cov, w)) 
                    return -(expected_return - (dynamic_lambda / 2.0) * portfolio_variance - (0.5 * np.sum(w**2)) - (turnover_penalty * np.sum(np.abs(w - current_weights_arr))))
                    
                # 🌟 Optimizer Constraints ขั้นเทพ (Max weight + Sector Cap + Turnover Cap)
                max_single_weight = 0.25 
                bounds = tuple((0.0, max_single_weight) for _ in range(num_assets))
                
                constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                
                # Hard Turnover Constraint (Max 30% shift per rebalance)
                max_turnover_cap = 0.30 
                constraints.append({'type': 'ineq', 'fun': lambda w: max_turnover_cap - np.sum(np.abs(w - current_weights_arr))})
                
                # Dynamic Sector Constraints
                unique_sectors = port_df['Sector'].unique()
                for sector in unique_sectors:
                    sec_indices = [i for i, t in enumerate(port_tickers) if port_df.loc[port_df['Ticker'] == t, 'Sector'].values[0] == sector]
                    constraints.append({'type': 'ineq', 'fun': lambda w, idx=sec_indices: max_sector_cap - np.sum(w[idx])})
                
                opt_result = minimize(neg_utility, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300})
                
                if opt_result.success: port_df['Target_%'] = port_df['Ticker'].map(dict(zip(port_tickers, opt_result.x))) * 100.0
                else: raise ValueError("Matrix Non-Convergence")
                    
            except Exception as e:
                st.warning(f"Fallback to Risk Parity due to: {e}")
                port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
                port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        else: 
            port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
            port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        # 🛑 RSI Mean Reversion Overlay (Institutional Logic ไม่แบนมั่วซั่วแล้ว)
        fomo_list = []
        for t in port_df['Ticker']:
            rsi_val = rsi_data.get(t, 50)
            if rsi_val > 75:
                # คำนวณ Penalty Factor: ยิ่ง RSI สูง ยิ่งโดนกดน้ำหนักลง (แต่ไม่เป็น 0)
                penalty_factor = max(0.2, 1.0 - ((rsi_val - 75) / 25))
                # นำ Penalty ไปคูณลดน้ำหนักเป้าหมาย
                port_df.loc[port_df['Ticker'] == t, 'Target_%'] *= penalty_factor
                fomo_list.append(f"{t} (ลดน้ำหนัก {(1 - penalty_factor)*100:.0f}%)")
        
        if fomo_list:
            # Re-normalize ให้สัดส่วนรวมกลับมาเป็น 100%
            if port_df['Target_%'].sum() > 0: 
                port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0
            fomo_msg = f"📉 **Mean Reversion Overlay:** ตรวจพบหุ้น Overbought ทำการลดเป้าหมายเพื่อคุมความเสี่ยง แต่ไม่ขัดขา Momentum -> {', '.join(fomo_list)}"

        # Execution Budget Math 
        target_total = total_port_value + actual_budget
        port_df['Target_Val'] = target_total * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current'] 
        port_df['Buy_Amount'] = 0.0
        port_df['Sell_Amount'] = 0.0
        
        buy_mask = port_df['Deficit'] > min_order_thb
        sum_deficit = port_df.loc[buy_mask, 'Deficit'].sum()
        if sum_deficit > 0:
            port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Deficit'] / sum_deficit) * actual_budget
            port_df['Buy_Amount'] = port_df['Buy_Amount'].apply(lambda x: round(x, 2) if x >= min_order_thb else 0.0)
        
        sell_mask = (port_df['Deficit'] < -min_order_thb) & (port_df['Weight_%'] > (max_sector_cap*100)/2)
        port_df.loc[sell_mask, 'Sell_Amount'] = port_df.loc[sell_mask, 'Deficit'].abs().round(2)
        
        cash_reserve = actual_budget - port_df['Buy_Amount'].sum()
        status_box.update(label="--- Quant Engine ประมวลผลเสร็จสิ้น ---", state="complete")
        
        # ==========================================
        # 🎨 การแสดงผลตารางใบสั่งซื้อพื้นฐาน
        # ==========================================
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ'})
        out['Thesis'] = out['หุ้น'].map(lambda x: THESIS_DB.get(x, "Quant Alpha"))
        out['MDD'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk).str.replace('ดอย (ซื้อระวัง)', 'ระวังดอย')
        out['รับ/ต้าน'] = out['หุ้น'].map(sr_data)
        out['ขาย'] = out['Sell_Amount'].fillna(0.0)
        for c in ['เป้า%', 'ทุนเดิม']: out[c] = out[c].round(2)
            
        st.markdown(f"### 📋 2. ตาราง Quant Allocation")
        if fomo_msg: st.warning(fomo_msg)
        display_cols = ['หุ้น', 'Thesis', 'MDD', 'RSI', 'รับ/ต้าน', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']
        st.dataframe(out[display_cols].sort_values(by='ซื้อ', ascending=False), use_container_width=True, hide_index=True)
        
        buy_list = out[out['ซื้อ'] > 0].sort_values(by='ซื้อ', ascending=False)
        proposed_buys_json = buy_list[['หุ้น', 'ซื้อ', 'Thesis']].to_dict('records')

        # ==========================================
        # 🤖 AI INSTITUTIONAL AUDIT ENGINE 
        # ==========================================
        st.markdown("---")
        st.markdown("### 🏛️ ระบบตรวจสอบโดย AI (Institutional Audit Engine)")
        
        if st.button("🧠 รันการตรวจสอบโครงสร้างพอร์ต (Run AI Audit)", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("❌ ไม่พบ API Key! โปรดใส่ GEMINI_API_KEY")
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        'gemini-3.1-flash-lite',
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    sector_totals = port_df.groupby('Sector')['Current'].sum().reset_index()
                    sector_exposure = [{"Sector": row['Sector'], "Weight_%": round((row['Current']/total_port_value)*100, 1) if total_port_value > 0 else 0} for _, row in sector_totals.iterrows()]
                    top_alpha = [{"Ticker": row['Ticker'], "Alpha_Score": round(row['Alpha_Score'], 2)} for _, row in final_df.head(5).iterrows()]
                    
                    port_state = {
                        "vix_level": round(vix_current, 1),
                        "market_regime": "PANIC" if is_panic else "NORMAL",
                        "proposed_buys": proposed_buys_json,
                        "current_sector_exposure": sector_exposure,
                        "top_alpha_candidates": top_alpha,
                        "constraints": f"Max Single Weight: 25%, Max Sector: {max_sector_cap*100}%"
                    }
                    port_state_str = json.dumps(port_state, indent=2)

                    board_container = st.container()
                    with board_container:
                        st.markdown("#### 📡 ตรวจพบช่องโหว่ความเสี่ยง (Audit JSON Log)")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            with st.spinner("Risk Auditor (Python + AI)..."):
                                prompt_cro = f"""
                                You are a strict Risk Audit system. Do not roleplay.
                                Objective: Audit the 'Portfolio State' for constraint violations. Explain anomalies.
                                Portfolio State: {port_state_str}
                                Return ONLY JSON:
                                {{
                                  "risk_level": "LOW", "MEDIUM", or "HIGH",
                                  "liquidity_concern": boolean,
                                  "concentration_anomaly_detected": boolean,
                                  "audit_explanation": "string"
                                }}
                                """
                                res_cro = model.generate_content(prompt_cro).text
                                cro_data = json.loads(res_cro)
                                st.json(cro_data)

                        with col2:
                            with st.spinner("Alpha Auditor (Python + AI)..."):
                                prompt_pm = f"""
                                You are an Alpha Validation system. Do not roleplay.
                                Objective: Verify if 'proposed_buys' aligns with high 'Alpha_Score' candidates. Provide reasoning.
                                Portfolio State: {port_state_str}
                                Return ONLY JSON:
                                {{
                                  "alpha_alignment_score": float (0.0 to 1.0),
                                  "missed_opportunities": ["Ticker1", "Ticker2"],
                                  "audit_explanation": "string"
                                }}
                                """
                                res_pm = model.generate_content(prompt_pm).text
                                pm_data = json.loads(res_pm)
                                st.json(pm_data)

                        st.markdown("---")
                        st.markdown("#### ⚖️ ระบบประเมินผลชี้ขาด (Python Governance Layer)")
                        
                        python_sector_violation = any(sec['Weight_%'] > (max_sector_cap*100) for sec in sector_exposure)
                        confidence = pm_data.get("alpha_alignment_score", 0.0)
                        
                        if python_sector_violation:
                            st.error(f"🔴 **EXECUTION BLOCKED (HOLD CASH)**\n\n**Hard Constraint Failed:** Python ตรวจพบการละเมิดเพดาน Sector > {max_sector_cap*100}%\n**AI Risk Notes:** {cro_data.get('audit_explanation')}")
                        elif confidence > 0.5:
                            st.success(f"🟢 **EXECUTION APPROVED**\n\n**Audit Passed:** โครงสร้างความเสี่ยงถูกต้องตามสมการ Scipy Minimize (Confidence: {confidence})\n**AI Notes:** {pm_data.get('audit_explanation')}")
                        else:
                            st.warning(f"🟡 **EXECUTION WARNING**\n\n**Low AI Alignment:** {confidence} - ระบบอาจเสนอซื้อหุ้นเชิง Defensive มากกว่า Alpha ตามสภาวะตลาด\n**AI Notes:** {pm_data.get('audit_explanation')}")
                            
                except json.JSONDecodeError:
                    st.error("❌ ขัดข้อง: AI ไม่ได้ตอบกลับมาเป็น JSON Format ที่ถูกต้อง")
                except Exception as e:
                    st.error(f"❌ ระบบประมวลผลขัดข้อง: {e}")
