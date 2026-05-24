import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os, sys
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.signal import argrelextrema 

warnings.filterwarnings("ignore")

from quant_math import calc_zscore, calculate_smart_weights, allocate_v21_fixed, calculate_rsi, check_doi_risk

def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

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
    except:
        return "-"

# ==========================================
# 💾 ฐานข้อมูลความจำ & หมวดหมู่
# ==========================================
PORTFOLIO_FILE = "my_portfolio_data.csv"

SECTOR_DB = {
    "💻 Tech": ["NVDA", "GOOG", "META", "CSCO", "TXN", "MSFT", "AAPL", "AMD", "PLTR", "AVGO"],
    "🛍️ Consumer": ["KO", "BLK", "MELI", "V", "MA", "WMT", "COST", "JPM"],
    "🩺 Health": ["JNJ", "ISRG", "LLY", "UNH", "ABBV"],
    "🚗 EV": ["TSLA", "UBER", "TM"],
    "🌿 Green": ["FSLR", "ENPH", "NEE", "SEDG"]
}
ticker_to_sector = {ticker: sector for sector, tickers in SECTOR_DB.items() for ticker in tickers}
DEFENSIVE_SECTORS = ["🛍️ Consumer", "🩺 Health"]

if 'dca_budget' not in st.session_state:
    st.session_state['dca_budget'] = 500.0
if 'min_order_thb' not in st.session_state:
    st.session_state['min_order_thb'] = 50.0

st.title("🛡️ HYBRID SENTINEL DCA (V.50.11 The Final Boss)")
st.markdown("ระบบจัดพอร์ตระดับ 10/10 **(ฉบับปิดจ็อบ: เพิ่ม Adaptive Lambda ปรับความกลัวตามชีพจรตลาด!)**")
st.markdown("---")

# ==========================================
# 🗂️ แถบด้านซ้าย (Sidebar)
# ==========================================
st.sidebar.subheader("🗂️ หุ้นในพอร์ตของคุณ")
tickers_input = st.sidebar.text_area("รายชื่อหุ้นที่ถืออยู่ (คั่นด้วยลูกน้ำ)", "FSLR, JNJ, KO, CSCO, TXN, V, NEE, NVDA")
my_portfolio = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.sidebar.markdown("---")

st.sidebar.subheader("🧠 เลือกสมองกลจัดพอร์ต")
engine_choice = st.sidebar.radio(
    "เข็มทิศการลงทุนประจำรอบ:",
    [
        "🧠 Auto-Pilot (Utility + BL + Adaptive Lambda)",
        "🛡️ Safe Mode (Risk Parity เน้นปลอดภัยสูงสุด)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ ปรับจูนความซิ่งตั้งต้น (Base Utility)")
base_lambda = st.sidebar.slider(
    "ระดับความกลัวตั้งต้น (Base Risk Aversion)", 
    min_value=0.1, max_value=10.0, value=2.0, step=0.1,
    help="ระบบจะนำค่านี้ไปคูณกับความผันผวนของตลาดแบบอัตโนมัติ"
)
st.sidebar.info("💡 **Active Modules:** \n- Black-Litterman Logic\n- Adaptive Lambda (Vol Ratio)\n- EWMA Semi-Cov (63D)\n- L2 & L1 Penalties")
st.sidebar.markdown("---")

# ==========================================
# 🖥️ หน้าจอหลัก (Main UI)
# ==========================================
ai_signals = st.session_state.get('ai_signals', {})

col_input1, col_input2 = st.columns(2)
with col_input1: 
    budget = st.number_input("💵 งบประมาณตั้งต้นรอบนี้ (บาท)", min_value=100, max_value=50000, value=int(st.session_state['dca_budget']), step=100)
    st.session_state['dca_budget'] = float(budget)
with col_input2: 
    min_order_thb = st.number_input("🚦 ยอดซื้อขั้นต่ำต่อหุ้น (บาท)", min_value=10, max_value=500, value=int(st.session_state['min_order_thb']), step=10)
    st.session_state['min_order_thb'] = float(min_order_thb)

st.markdown("### 💼 1. ตรวจสอบพอร์ตปัจจุบัน")

saved_dict = {}
if os.path.exists(PORTFOLIO_FILE):
    try:
        saved_df = pd.read_csv(PORTFOLIO_FILE)
        if not saved_df.empty and "รายชื่อหุ้น" in saved_df.columns:
            saved_df["ยอดเงินปัจจุบัน (บาท)"] = pd.to_numeric(saved_df["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0.0)
            saved_dict = dict(zip(saved_df["รายชื่อหุ้น"], saved_df["ยอดเงินปัจจุบัน (บาท)"]))
    except Exception:
        pass

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

sniper_msg = ""
bl_msg = ""
lambda_msg = ""
actual_budget = budget 

if st.button("🚀 รันระบบ Ultimate Rebalancer"):
    if not my_portfolio: 
        st.error("⚠️ โปรดระบุชื่อหุ้นที่แถบด้านข้างซ้ายมือก่อนครับ")
    else:
        status_box = st.status(f"🔮 เดินเครื่องควอนตัม: ผสานตลาด, AI, และความกลัวอัตโนมัติ...", expanded=True)
        
        benchmark = 'VOO'
        fx_ticker = 'USDTHB=X'
        
        block_print()
        market_data = yf.download([benchmark, fx_ticker], period="3y", progress=False)['Close']
        enable_print()
        
        # 🌟 V.50.11 The Final Boss: ตรวจจับชีพจรตลาดเพื่อปรับ Lambda อัตโนมัติ
        voo_ret = market_data[benchmark].pct_change().dropna()
        vol_30d = voo_ret.tail(30).std() * np.sqrt(252)
        vol_252d = voo_ret.tail(252).std() * np.sqrt(252)
        
        vol_ratio = vol_30d / vol_252d if vol_252d > 0 else 1.0
        dynamic_lambda = base_lambda * vol_ratio
        
        if vol_ratio > 1.2:
            lambda_msg = f"🌩️ **Adaptive Risk:** ตลาดผันผวนจัด (Ratio: {vol_ratio:.2f}) ระบบเร่งสวิตช์ความกลัว (λ) อัตโนมัติเป็น {dynamic_lambda:.2f}"
        elif vol_ratio < 0.8:
            lambda_msg = f"☀️ **Adaptive Risk:** ตลาดนิ่งสงบ (Ratio: {vol_ratio:.2f}) ระบบลดสวิตช์ความกลัว (λ) ลงเหลือ {dynamic_lambda:.2f} เพื่อเร่งทำกำไร!"
        else:
            lambda_msg = f"⚖️ **Adaptive Risk:** ตลาดปกติ (Ratio: {vol_ratio:.2f}) ใช้ค่าความกลัว (λ) ที่ {dynamic_lambda:.2f}"

        sma200_voo = market_data[benchmark].rolling(200).mean().iloc[-1]
        is_market_crashing = market_data[benchmark].iloc[-1] < sma200_voo
            
        market_proxy = [t for sublist in SECTOR_DB.values() for t in sublist]
        universe_list = list(set(my_portfolio + market_proxy)) 
        
        block_print()
        prices_3y = yf.download(universe_list, period="3y", progress=False)['Close']
        enable_print()
        
        prices_1y = prices_3y.tail(252) 
        returns_1y = prices_1y.pct_change().dropna()
        
        if returns_1y.empty:
            status_box.update(label="❌ ดึงข้อมูลล้มเหลว โปรดลองใหม่อีกครั้ง", state="error")
            st.stop()
            
        ann_ret = returns_1y.mean() * 252
        ann_vol = returns_1y.std() * np.sqrt(252)
        sharpe_ratio = ann_ret / ann_vol
        
        downside_returns = returns_1y.copy()
        downside_returns[downside_returns > 0] = 0
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = ann_ret / downside_vol
        
        ret_3m = prices_1y.pct_change(periods=63).iloc[-1]
        ret_6m = prices_1y.pct_change(periods=126).iloc[-1]
        avg_mom = (ret_3m + ret_6m) / 2
        vol = prices_1y.pct_change().tail(126).std() * np.sqrt(252)
        sma_200 = prices_1y.rolling(window=200).mean().iloc[-1]
        
        df = pd.DataFrame({'Ticker': prices_1y.columns, 'Raw_Mom': (avg_mom / vol), 'Price': prices_1y.iloc[-1], 'SMA_200': sma_200}).dropna().reset_index(drop=True)
        
        quality_data, rsi_data, sr_data = [], {}, {}
        for t in df['Ticker'].tolist():
            roa, margin = None, None
            for attempt in range(2):
                try:
                    block_print()
                    info = yf.Ticker(t).info
                    enable_print()
                    roa, margin = info.get('returnOnAssets'), info.get('profitMargins')
                    if roa is not None: break
                    time.sleep(0.01)
                except:
                    enable_print()
                    continue
            quality_data.append({'Ticker': t, 'ROA': roa * 100 if roa is not None else np.nan, 'Margin': margin * 100 if margin is not None else np.nan})
            
            series_clean = prices_1y[t].dropna()
            rsi_data[t] = calculate_rsi(series_clean).iloc[-1] if len(series_clean) > 14 else 50.0
            sr_data[t] = find_sr_levels(series_clean)
        
        final_df = pd.merge(df, pd.DataFrame(quality_data), on='Ticker')
        final_df['Alpha_Score'] = (calc_zscore(final_df['Raw_Mom']) * 0.5) + (calc_zscore(final_df['ROA'].fillna(final_df['ROA'].median())) * 0.25) + (calc_zscore(final_df['Margin'].fillna(final_df['Margin'].median())) * 0.25)
        
        final_df['Sharpe'] = final_df['Ticker'].map(sharpe_ratio)
        final_df['Sortino'] = final_df['Ticker'].map(sortino_ratio)
        
        final_df = final_df.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
        
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        port_df['Sector'] = port_df['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0

        # ==========================================
        # ⚖️ THE FINAL BOSS ENGINE 
        # ==========================================
        if "Auto-Pilot" in engine_choice:
            status_box.update(label=f"🧮 เดินเครื่องสมการ: BL + Adaptive Lambda...")
            port_tickers = port_df['Ticker'].tolist()
            port_returns = returns_1y[port_tickers]
            num_assets = len(port_tickers)
            
            try:
                downside_port_returns = port_returns.copy()
                downside_port_returns[downside_port_returns > 0] = 0
                last_date = downside_port_returns.index[-1]
                ewma_cov_df = downside_port_returns.ewm(span=63).cov().loc[last_date] * 252
                ewma_cov = ewma_cov_df.values 
                
                inv_vol = 1.0 / returns_1y[port_tickers].std()
                w_eq = (inv_vol / inv_vol.sum()).values
                # ใช้ dynamic_lambda แทนที่ของเดิม
                Pi = dynamic_lambda * np.dot(ewma_cov, w_eq) 
                
                tau = 0.05
                Q = np.zeros(num_assets)
                P = np.eye(num_assets) 
                omega_diag = np.zeros(num_assets)
                
                ai_active_count = 0
                sniper_targets = []

                for i, t in enumerate(port_tickers):
                    series = prices_1y[t].dropna()
                    if len(series) >= 20:
                        sma20 = series.rolling(20).mean().iloc[-1]
                        std20 = series.rolling(20).std().iloc[-1]
                        lower_band = sma20 - (2 * std20)
                        current_p = series.iloc[-1]
                        rsi_val = rsi_data.get(t, 50.0)
                        
                        if current_p <= lower_band or rsi_val <= 40: 
                            sniper_targets.append(t)
                            Q[i] = Pi[i] + 0.20 
                            omega_diag[i] = ewma_cov[i, i] * tau * 0.1 
                            continue

                    if t in ai_signals:
                        ai_active_count += 1
                        ai_pred = (ai_signals[t].get('target_pct', 0) / 100) * 25.2 
                        acc = ai_signals[t].get('accuracy', 50.0)
                        Q[i] = ai_pred
                        omega_diag[i] = (ewma_cov[i, i] * tau) / max((acc/100), 0.01)
                    else:
                        Q[i] = Pi[i] 
                        omega_diag[i] = ewma_cov[i, i] * tau * 100 

                if len(sniper_targets) > 0:
                    sniper_msg = f"🎯 **Sniper Alert:** พบเป้าหมาย {', '.join(sniper_targets)} ทฤษฎีเบส์ได้ปรับดุลยภาพเพื่อช้อนซื้อแล้ว!"
                if ai_active_count > 0:
                    bl_msg = f"🤖 **Black-Litterman Active:** ผสานข้อมูล AI ({ai_active_count} ตัว) เข้ากับดุลยภาพตลาดสำเร็จ!"

                Omega = np.diag(omega_diag)
                
                tau_cov_inv = np.linalg.inv(tau * ewma_cov)
                omega_inv = np.linalg.inv(Omega)
                
                term1 = np.linalg.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
                term2 = np.dot(tau_cov_inv, Pi) + np.dot(np.dot(P.T, omega_inv), Q)
                mu_bl = np.dot(term1, term2) 

                current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(num_assets)
                gamma_penalty = 0.5   
                tau_penalty = 0.02    

                def neg_utility(w):
                    expected_return = np.sum(mu_bl * w)
                    portfolio_variance = np.dot(w.T, np.dot(ewma_cov, w)) 
                    l2_penalty = gamma_penalty * np.sum(w**2)
                    l1_turnover_penalty = tau_penalty * np.sum(np.abs(w - current_weights_arr))
                    
                    # 🌟 ใช้ dynamic_lambda แทน
                    utility = expected_return - (dynamic_lambda / 2.0) * portfolio_variance - l2_penalty - l1_turnover_penalty
                    return -utility
                    
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bounds = tuple((0.0, 1.0) for _ in range(num_assets))
                init_guess = num_assets * [1. / num_assets]
                
                opt_result = minimize(neg_utility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
                
                if opt_result.success:
                    opt_weights_dict = dict(zip(port_tickers, opt_result.x))
                    port_df['Target_%'] = port_df['Ticker'].map(opt_weights_dict) * 100.0
                else:
                    raise ValueError("Matrix Non-Convergence")
                    
            except Exception as e:
                st.warning(f"⚠️ ระบบเบส์ซับซ้อนเกินไป สลับไปใช้ 'Risk Parity' ชั่วคราวครับ! ({e})")
                port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
                port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        else: 
            status_box.update(label=f"🛡️ กำลังคำนวณน้ำหนักเกราะป้องกัน (Risk Parity)...")
            port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
            port_df.loc[(port_df['Rank'] > 15) & (port_df['Price'] < port_df['SMA_200']), 'Inv_Vol'] = 0.0
            port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        sector_totals = port_df.groupby('Sector')['Current'].sum().reset_index()
        overweight_sectors = []
        max_sector_cap = 40 if is_market_crashing else 60 
        
        for _, row in sector_totals.iterrows():
            sec_weight = (row['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
            if sec_weight > max_sector_cap: overweight_sectors.append(row['Sector'])
            
        if overweight_sectors:
            port_df.loc[port_df['Sector'].isin(overweight_sectors), 'Target_%'] = 0.0
            sum_valid_target = port_df['Target_%'].sum()
            if sum_valid_target > 0:
                port_df['Target_%'] = (port_df['Target_%'] / sum_valid_target) * 100.0
        # ==========================================
        # 🛑 FOMO CIRCUIT BREAKER (Anti-Doi System)
        # ==========================================
        fomo_list = [t for t in port_df['Ticker'] if rsi_data.get(t, 50) > 75] 
        if fomo_list:
            port_df.loc[port_df['Ticker'].isin(fomo_list), 'Target_%'] = 0.0
            sum_target = port_df['Target_%'].sum()
            if sum_target > 0:
                port_df['Target_%'] = (port_df['Target_%'] / sum_target) * 100.0
            st.warning(f"🛑 **FOMO Breaker ทำงาน:** เบรกหัวทิ่ม! พบหุ้น Overbought กราฟตึงจัด ({', '.join(fomo_list)}) ระบบสั่งระงับการซื้อชั่วคราวเพื่อป้องกันการติดดอย!")
        # === จัดการตังค์ ===
        target_total = total_port_value + actual_budget
        port_df['Target_Val'] = target_total * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current'] 
        port_df['Buy_Amount'] = 0.0
        port_df['Sell_Amount'] = 0.0
        
        anchor_ticker = None
        if total_port_value > 0:
            anchor_ticker = port_df.sort_values('Current', ascending=False).iloc[0]['Ticker']
            corr_with_anchor = returns_1y.corr()[anchor_ticker]
            port_df['Corr_Multiplier'] = port_df['Ticker'].map(lambda x: 1 - (corr_with_anchor.get(x, 0) * 0.5))
        else:
            port_df['Corr_Multiplier'] = 1.0

        buy_mask = port_df['Deficit'] > min_order_thb
        sum_deficit = port_df.loc[buy_mask, 'Deficit'].sum()
        
        if sum_deficit > 0:
            port_df.loc[buy_mask, 'Adj_Deficit'] = port_df.loc[buy_mask, 'Deficit'] * port_df.loc[buy_mask, 'Corr_Multiplier']
            sum_adj_deficit = port_df.loc[buy_mask, 'Adj_Deficit'].sum()
            if sum_adj_deficit > 0:
                port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Adj_Deficit'] / sum_adj_deficit) * actual_budget
            
            port_df['Buy_Amount'] = port_df['Buy_Amount'].apply(lambda x: round(x, 2) if x >= min_order_thb else 0.0)
        
        sell_mask = (port_df['Deficit'] < -min_order_thb) & (port_df['Weight_%'] > max_sector_cap/2)
        port_df.loc[sell_mask, 'Sell_Amount'] = port_df.loc[sell_mask, 'Deficit'].abs().round(2)
        
        cash_reserve = actual_budget - port_df['Buy_Amount'].sum()

        status_box.update(label="--- คำนวณเสร็จสิ้น! ---", state="complete")
        
        # ==========================================
        # 🎨 การแสดงผลตาราง
        # ==========================================
        st.markdown("### 🍩 2. โครงสร้างความเสี่ยง (Guardrails & Correlation)")
        
        valid_port_tickers = [t for t in my_portfolio if t in prices_1y.columns]
        corr_matrix = returns_1y[valid_port_tickers].corr().round(2)
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig_pie = px.pie(sector_totals, values='Current', names='Sector', hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma)
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            if overweight_sectors:
                st.error(f"🛑 **HARD CAP:** กลุ่ม **{', '.join(overweight_sectors)}** บวมเกินเพดาน {max_sector_cap}%!")
            else:
                st.success(f"✅ น้ำหนักกระจายตัวสวยงาม (เพดานปัจจุบัน: {max_sector_cap}%)")
                
        with c2:
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
            fig_corr.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=280, title="เรดาร์ความสัมพันธ์ (Correlation Matrix)")
            st.plotly_chart(fig_corr, use_container_width=True)
                
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ'})
        
        out['สถานะ'] = '🟢 Elite'
        out.loc[(out['Rank'] > 15) & (out['Price'] < out['SMA_200']), 'สถานะ'] = '🔴 Sell (หลุดเทรนด์)'
        
        is_defensive = out['Sector'].isin(DEFENSIVE_SECTORS)
        out.loc[is_defensive & (out['สถานะ'].str.contains('Sell')), 'สถานะ'] = '🟢 ถือรับปันผล (กองหลัง)'
        
        out['ขาย'] = out['Sell_Amount']
        out.loc[out['ขาย'] > 0, 'สถานะ'] = '🔥 รินขาย (Rebalance)'
        
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk).str.replace('ดอย (ซื้อระวัง)', 'ระวังดอย').str.replace('ของถูก (เก็บสะสม)', 'ของถูก')
        out.loc[is_defensive & out['RSI'].str.contains('ระวังดอย'), 'RSI'] = '📈 รันเทรนด์ (ไม่ต้องตกใจ)'
        
        out['รับ/ต้าน (S1, S2 | R1)'] = out['หุ้น'].map(sr_data)
        out['คูณเงิน (Corr)'] = out['Corr_Multiplier'].apply(lambda x: f"x{x:.2f}") if 'Corr_Multiplier' in out.columns else "-"
        
        for c in ['เป้า%', 'ทุนเดิม']: out[c] = out[c].round(2)
            
        st.markdown(f"### 📋 3. ตารางใบสั่งซื้ออัจฉริยะ (The Final Auto-Pilot)")
        
        if lambda_msg != "":
            st.info(lambda_msg) # แสดงแจ้งเตือนความกลัวของ AI
        if sniper_msg != "":
            st.success(sniper_msg)
        if bl_msg != "":
            st.info(bl_msg)
            
        display_cols = ['หุ้น', 'สถานะ', 'รับ/ต้าน (S1, S2 | R1)', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']
        st.dataframe(out[display_cols].sort_values(by='ซื้อ', ascending=False), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        c_buy, c_sell = st.columns(2)
        buy_list = out[out['ซื้อ'] > 0].sort_values(by='ซื้อ', ascending=False)
        sell_list = out[out['ขาย'] > 0].sort_values(by='ขาย', ascending=False)
        
        with c_buy:
            st.markdown("### 🛒 โพยสั่งซื้อ (Dime!)")
            text_buy = f"📅 งบซื้อรอบนี้ ({actual_budget:,.0f} บ.)\n----------------\n"
            for _, row in buy_list.iterrows(): text_buy += f"🔹 {row['หุ้น']} = ซื้อ {row['ซื้อ']} บ.\n"
            if cash_reserve > 0: text_buy += f"💰 พักเงินสด = {cash_reserve:.2f} บ.\n"
            st.code(text_buy, language="text")
            
        with c_sell:
            st.markdown("### 💰 โพยรินขาย (Rebalance)")
            text_sell = f"📅 คำสั่งขายปรับสมดุลพอร์ต\n----------------\n"
            if sell_list.empty: text_sell += "✅ ไม่มีหุ้นที่ล้นพอร์ต\n"
            else:
                for _, row in sell_list.iterrows(): text_sell += f"🔻 {row['หุ้น']} = ขาย {row['ขาย']} บ.\n"
            st.code(text_sell, language="text")

        # ==========================================
        # 🏆 🌟 เรดาร์สแกนหุ้นจ่าฝูง (TOP 20 ALPHA LEADERBOARD)
        # ==========================================
        st.markdown("---")
        st.subheader("🏆 📡 [RADAR] TOP 20 ALPHA (Risk-Adjusted)")
        
        top20_df = final_df.head(20).copy()
        top20_df['Sector'] = top20_df['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        top20_df['Alpha_Score'] = top20_df['Alpha_Score'].round(2)
        top20_df['Sharpe'] = top20_df['Sharpe'].round(2)
        top20_df['Sortino'] = top20_df['Sortino'].round(2)
        top20_df['สถานะ'] = top20_df['Ticker'].apply(lambda x: "💼 ถืออยู่" if x in my_portfolio else "✨ เป้าหมายใหม่")
        
        display_top20 = top20_df[['Rank', 'Ticker', 'Sector', 'Alpha_Score', 'Sharpe', 'Sortino', 'สถานะ']].rename(
            columns={'Ticker': 'หุ้น', 'Alpha_Score': 'Alpha', 'Sharpe': 'Sharpe Ratio', 'Sortino': 'Sortino Ratio'}
        )
        st.dataframe(display_top20, use_container_width=True, hide_index=True)

        # ==========================================
        # ⏱️ THE TIME MACHINE 
        # ==========================================
        st.markdown("---")
        with st.expander("⏱️ [TIME MACHINE] สถิติโลกความจริงย้อนหลัง 3 ปี (หักค่าธรรมเนียมแล้ว)", expanded=False):
            try:
                bt_data = prices_3y[my_portfolio].copy()
                bt_data['VOO'] = market_data['VOO']
                bt_data = bt_data.dropna()
                
                daily_ret = bt_data.pct_change().dropna()
                
                weights = []
                valid_tickers = []
                for t in my_portfolio:
                    if t in bt_data.columns and t in port_df['Ticker'].values:
                        w = port_df.loc[port_df['Ticker']==t, 'Target_%'].values[0]
                        weights.append(w)
                        valid_tickers.append(t)
                
                sum_w = sum(weights)
                if sum_w > 0: weights = np.array(weights) / sum_w
                else: weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
                
                friction_drag_daily = 0.0020 / 252 
                
                port_daily_ret = (daily_ret[valid_tickers] * weights).sum(axis=1) - friction_drag_daily
                port_cum = (1 + port_daily_ret).cumprod() * 100
                voo_cum = (1 + daily_ret['VOO']).cumprod() * 100
                
                plot_df = pd.DataFrame({'พอร์ตของเรา (หักต้นทุน)': port_cum, 'ตลาดโลก S&P 500 (VOO)': voo_cum})
                
                fig_bt = px.line(plot_df, labels={'value': 'การเติบโต (จุดเริ่มต้น = 100)', 'Date': 'วันที่'},
                                 color_discrete_sequence=['#00d4ff', '#ff4b4b'])
                fig_bt.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                
                st.plotly_chart(fig_bt, use_container_width=True)
                
                years_passed = len(daily_ret) / 252
                port_cagr = ((port_cum.iloc[-1] / 100) ** (1/years_passed)) - 1
                voo_cagr = ((voo_cum.iloc[-1] / 100) ** (1/years_passed)) - 1
                
                col_bt1, col_bt2 = st.columns(2)
                col_bt1.metric("CAGR พอร์ตเรา (เฉลี่ยต่อปี)", f"{port_cagr*100:.2f}%")
                col_bt2.metric("CAGR ตลาด S&P 500", f"{voo_cagr*100:.2f}%")
                
            except Exception as e:
                st.error(f"ข้อมูลหุ้นบางตัวมีประวัติสั้นกว่า 3 ปี ทำให้จำลองผลระยะยาวไม่ได้ครับ ({e})")
