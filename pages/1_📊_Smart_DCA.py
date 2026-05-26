import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os, sys
import plotly.express as px
from scipy.optimize import minimize
from scipy.signal import argrelextrema 

warnings.filterwarnings("ignore")

# ==========================================
# 🛠️ ฟังก์ชันคณิตศาสตร์และเครื่องมือ Quant
# ==========================================
def calc_zscore(series): 
    if series.std() == 0: return series - series.mean()
    return (series - series.mean()) / series.std()

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
# 🧠 อัปเกรด: สร้างชิปความจำให้ระบบ
if 'system_run' not in st.session_state: st.session_state['system_run'] = False

st.set_page_config(page_title="QuantHQ DCA", page_icon="🛡️", layout="wide")
st.title("🛡️ QUANT-HQ DCA (V.50.19 Anti-Amnesia AI)")
st.markdown("ระบบจัดพอร์ตระดับสถาบัน **(ผสาน AI คุมประพฤติโพยสั่งซื้อ 100% Stable)**")
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

# ==========================================
# 🖥️ หน้าจอหลัก (Main UI)
# ==========================================
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

sniper_msg, bl_msg, lambda_msg, fomo_msg = "", "", "", ""
actual_budget = budget 

# 🧠 อัปเกรด: บันทึกความจำลงชิป
if st.button("🚀 รันระบบ Ultimate Rebalancer", type="primary"):
    st.session_state['system_run'] = True

if st.session_state['system_run']:
    if not my_portfolio: 
        st.error("⚠️ โปรดระบุชื่อหุ้นก่อนครับ")
        st.session_state['system_run'] = False
    else:
        status_box = st.status(f"🔮 เดินเครื่องสมองกลประมวลผล Matrix สากล...", expanded=False)
        
        benchmark = 'VOO'
        vix_ticker = '^VIX'
        
        block_print()
        market_data = yf.download([benchmark, vix_ticker], period="3y", progress=False)['Close']
        enable_print()
        
        vix_current = market_data[vix_ticker].iloc[-1] if vix_ticker in market_data else 20.0
        sma200_voo = market_data[benchmark].rolling(200).mean().iloc[-1]
        is_market_crashing = market_data[benchmark].iloc[-1] < sma200_voo
        
        is_panic = False
        if vix_current > 25:
            st.error(f"🚨 **Market Regime: PANIC (VIX = {vix_current:.1f})** ตลาดเกิดความกลัวรุนแรง! ดึงเกราะความเสี่ยงขึ้น")
            is_panic = True
        elif vix_current < 15:
            st.success(f"🐂 **Market Regime: BULL (VIX = {vix_current:.1f})** ตลาดกระทิงนิ่งสงบ! ขยายเพดานสายบุกสุดตัว")
        else:
            st.info(f"⚖️ **Market Regime: NORMAL (VIX = {vix_current:.1f})** ตลาดทำงานในสภาวะปกติ")

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
        
        ann_ret = returns_1y.mean() * 252
        ann_vol = returns_1y.std() * np.sqrt(252)
        sharpe_ratio = ann_ret / ann_vol
        downside_returns = returns_1y.copy()
        downside_returns[downside_returns > 0] = 0
        sortino_ratio = ann_ret / (downside_returns.std() * np.sqrt(252))
        
        ret_3m = prices_1y.pct_change(periods=63).iloc[-1]
        ret_6m = prices_1y.pct_change(periods=126).iloc[-1]
        avg_mom = (ret_3m + ret_6m) / 2
        vol = prices_1y.pct_change().tail(126).std() * np.sqrt(252)
        
        df = pd.DataFrame({'Ticker': prices_1y.columns, 'Raw_Mom': (avg_mom / vol), 'Price': prices_1y.iloc[-1]}).dropna().reset_index(drop=True)
        
        quality_data, rsi_data, sr_data = [], {}, {}
        for t in df['Ticker'].tolist():
            roa, margin = None, None
            try:
                block_print()
                info = yf.Ticker(t).info
                enable_print()
                roa, margin = info.get('returnOnAssets'), info.get('profitMargins')
            except: enable_print()
            quality_data.append({'Ticker': t, 'ROA': roa * 100 if roa is not None else np.nan, 'Margin': margin * 100 if margin is not None else np.nan})
            series_clean = prices_1y[t].dropna()
            rsi_data[t] = calculate_rsi(series_clean).iloc[-1] if len(series_clean) > 14 else 50.0
            sr_data[t] = find_sr_levels(series_clean)
            
        final_df = pd.merge(df, pd.DataFrame(quality_data), on='Ticker')
        final_df['Alpha_Score'] = (calc_zscore(final_df['Raw_Mom']) * 0.5) + (calc_zscore(final_df['ROA'].fillna(final_df['ROA'].median())) * 0.25) + (calc_zscore(final_df['Margin'].fillna(final_df['Margin'].median())) * 0.25)
        final_df['Max_Drawdown'] = final_df['Ticker'].map(max_dd)
        
        final_df = final_df.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
        
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        port_df['Sector'] = port_df['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0

        # ==========================================
        # ⚖️ THE ENGINE: BLACK-LITTERMAN MATRIX 
        # ==========================================
        if "Auto-Pilot" in engine_choice:
            status_box.update(label=f"🧮 กำลังแก้สมการอนุพันธ์เพื่อค้นหาจุดดุลยภาพความเสี่ยง...")
            port_tickers = port_df['Ticker'].tolist()
            port_returns = returns_1y[port_tickers]
            num_assets = len(port_tickers)
            
            try:
                downside_port_returns = port_returns.copy()
                downside_port_returns[downside_port_returns > 0] = 0
                last_date = downside_port_returns.index[-1]
                ewma_cov = downside_port_returns.ewm(span=63).cov().loc[last_date].values * 252
                
                inv_vol = 1.0 / returns_1y[port_tickers].std()
                w_eq = (inv_vol / inv_vol.sum()).values
                Pi = dynamic_lambda * np.dot(ewma_cov, w_eq) 
                
                tau = 0.05
                Q = np.zeros(num_assets)
                P = np.eye(num_assets) 
                omega_diag = np.zeros(num_assets)
                
                sniper_targets = []
                for i, t in enumerate(port_tickers):
                    series = prices_1y[t].dropna()
                    if len(series) >= 20:
                        lower_band = series.rolling(20).mean().iloc[-1] - (2 * series.rolling(20).std().iloc[-1])
                        if series.iloc[-1] <= lower_band or rsi_data.get(t, 50.0) <= 40: 
                            sniper_targets.append(t)
                            Q[i] = Pi[i] + 0.20 
                            omega_diag[i] = ewma_cov[i, i] * tau * 0.1 
                            continue
                    Q[i] = Pi[i] 
                    omega_diag[i] = ewma_cov[i, i] * tau * 100 

                if sniper_targets: sniper_msg = f"🎯 **Sniper Alert:** พบหุ้นดิ่งแตะแนวรับ {', '.join(sniper_targets)} สั่งปรับน้ำหนักเพื่อเข้าช้อนซื้อ!"

                Omega = np.diag(omega_diag)
                tau_cov_inv = np.linalg.inv(tau * ewma_cov)
                omega_inv = np.linalg.inv(Omega)
                
                term1 = np.linalg.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
                term2 = np.dot(tau_cov_inv, Pi) + np.dot(np.dot(P.T, omega_inv), Q)
                mu_bl = np.dot(term1, term2) 

                current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(num_assets)
                
                def neg_utility(w):
                    expected_return = np.sum(mu_bl * w)
                    portfolio_variance = np.dot(w.T, np.dot(ewma_cov, w)) 
                    return -(expected_return - (dynamic_lambda / 2.0) * portfolio_variance - (0.5 * np.sum(w**2)) - (0.02 * np.sum(np.abs(w - current_weights_arr))))
                    
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bounds = tuple((0.0, 1.0) for _ in range(num_assets))
                opt_result = minimize(neg_utility, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
                
                if opt_result.success: port_df['Target_%'] = port_df['Ticker'].map(dict(zip(port_tickers, opt_result.x))) * 100.0
                else: raise ValueError("Matrix Non-Convergence")
                    
            except Exception as e:
                st.warning(f"⚠️ ตลาดผันผวนเกินขนาด สลับไปใช้กลไก Risk Parity คุมกระดูกพอร์ตอัตโนมัติ! ({e})")
                port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
                port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        else: 
            status_box.update(label=f"🛡️ ดำเนินงานจัดสัดส่วนแบบกระจายความเสี่ยงขั้นสูงสุด (Risk Parity)...")
            port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
            port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        sector_totals = port_df.groupby('Sector')['Current'].sum().reset_index()
        overweight_sectors = []
        max_sector_cap = 50 if is_panic else (55 if is_market_crashing else 70) 
        
        for _, row in sector_totals.iterrows():
            sec_weight = (row['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
            if sec_weight > max_sector_cap: overweight_sectors.append(row['Sector'])
            
        if overweight_sectors:
            port_df.loc[port_df['Sector'].isin(overweight_sectors), 'Target_%'] = 0.0
            if port_df['Target_%'].sum() > 0: port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0

        # FOMO CIRCUIT BREAKER
        fomo_list = [t for t in port_df['Ticker'] if rsi_data.get(t, 50) > 75] 
        if fomo_list:
            port_df.loc[port_df['Ticker'].isin(fomo_list), 'Target_%'] = 0.0
            if port_df['Target_%'].sum() > 0: port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0
            fomo_msg = f"🛑 **Anti-FOMO Active:** สั่งระงับงบซื้อเพิ่มใน {', '.join(fomo_list)} (เป้า 0%) เพราะราคาตึงเสี่ยงติดดอย!"

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
        
        sell_mask = (port_df['Deficit'] < -min_order_thb) & (port_df['Weight_%'] > max_sector_cap/2)
        port_df.loc[sell_mask, 'Sell_Amount'] = port_df.loc[sell_mask, 'Deficit'].abs().round(2)
        
        cash_reserve = actual_budget - port_df['Buy_Amount'].sum()
        status_box.update(label="--- คำนวณสมการโครงสร้างเสร็จสิ้นแล้ว! ---", state="complete")
        
        # ==========================================
        # 🎨 การแสดงผลหน้าจอ
        # ==========================================
        st.markdown("### 🍩 2. โครงสร้างความเสี่ยง (Guardrails & Correlation)")
        valid_port_tickers = [t for t in my_portfolio if t in prices_1y.columns]
        corr_matrix = returns_1y[valid_port_tickers].corr().round(2)
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig_pie = px.pie(sector_totals, values='Current', names='Sector', hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma)
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
            fig_corr.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=280, title="เรดาร์ความสัมพันธ์ (Correlation Matrix)")
            st.plotly_chart(fig_corr, use_container_width=True)
                
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ'})
        out['Thesis'] = out['หุ้น'].map(lambda x: THESIS_DB.get(x, "Quant Alpha"))
        out['MDD'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk).str.replace('ดอย (ซื้อระวัง)', 'ระวังดอย')
        out['รับ/ต้าน'] = out['หุ้น'].map(sr_data)
        out['ขาย'] = out['Sell_Amount'].fillna(0.0)
        for c in ['เป้า%', 'ทุนเดิม']: out[c] = out[c].round(2)
            
        st.markdown(f"### 📋 3. ตารางใบสั่งซื้ออัจฉริยะ (The Final Auto-Pilot)")
        if lambda_msg != "": st.info(lambda_msg) 
        if fomo_msg != "": st.error(fomo_msg)
        if sniper_msg != "": st.success(sniper_msg)
            
        display_cols = ['หุ้น', 'Thesis', 'MDD', 'RSI', 'รับ/ต้าน', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']
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
            if sell_list.empty: text_sell += "✅ ไม่มีหุ้นที่ล้นพอร์ตจนเกิดเกณฑ์คุมความเสี่ยง\n"
            else:
                for _, row in sell_list.iterrows(): text_sell += f"🔻 {row['หุ้น']} = ขาย {row['ขาย']} บ.\n"
            st.code(text_sell, language="text")

       # ==========================================
        # 🤖 THE AI TRADE REVIEWER (V.10/10 Full Context)
        # ==========================================
        st.markdown("---")
        st.markdown("### 🤖 ให้ AI ตรวจทานความเสี่ยง (CRO) ก่อนโอนเงิน")
        
        if st.button("🔍 ส่งโพยให้ Chief Risk Officer ตรวจสอบ", type="secondary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("❌ ไม่พบ API Key! โปรดใส่ GEMINI_API_KEY ใน Settings -> Secrets")
            else:
                with st.spinner("CRO กำลังสแกนความเสี่ยงของพอร์ตโดยรวม..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-3.1-flash-lite')
                        
                        # 1. ดึงข้อมูล "โพยสั่งซื้อใหม่"
                        buy_str = "\n".join([f"- {row['หุ้น']}: ซื้อ {row['ซื้อ']} บาท ({row['Thesis']})" for _, row in buy_list.iterrows()])
                        if not buy_str: buy_str = "- ไม่มีคำสั่งซื้อในรอบนี้ (ระบบสั่งพักเงินสด)"
                        
                        # 2. ดึงข้อมูล "พอร์ตเดิมที่มีอยู่" (นี่คือจุดที่ทำให้ได้ 10/10)
                        current_port_str = ", ".join([f"{row['หุ้น']} ({row['ทุนเดิม']} บ.)" for _, row in out.iterrows() if row['ทุนเดิม'] > 0])
                        if not current_port_str: current_port_str = "พอร์ตว่างเปล่า (เพิ่งเริ่มตั้งต้น)"
                        
                        # 3. วิศวกรรมคำสั่ง 10/10 
                        # ⚔️ วิศวกรรมคำสั่ง 10/10 (นโยบายพอร์ต: Aggressive Growth เน้นบุก!)
                        prompt = f"""
                        ในฐานะ Chief Risk Officer (CRO) ประจำกองทุน Quant ที่มีนโยบายพอร์ตแบบ "Aggressive Growth (เน้นบุก ล่า Alpha)"
                        จงตรวจสอบโพยคำสั่งซื้อ DCA ประจำงวด ก่อนที่ผู้จัดการกองทุนจะส่งคำสั่งเข้าตลาด
                        
                        📊 [ข้อมูลหน้าตัก (Context)]
                        - นโยบายหลัก (Mandate): เน้นบุกหนัก (High-Beta / Compounders) ยอมรับความผันผวนได้สูงมาก
                        - งบประมาณรอบนี้: {actual_budget} บาท (เงื่อนไขโลกความจริง: ซื้อขั้นต่ำ 50 บาท/หุ้น)
                        - สภาวะตลาด (Regime): ดัชนีความกลัว VIX = {vix_current:.1f}
                        - หุ้นที่มีอยู่ในพอร์ตตอนนี้: {current_port_str}
                        
                        🛒 [โพยคำสั่งซื้อจากระบบสมองกล (Proposed Trades)]
                        {buy_str}
                        
                        ⚠️ [คำสั่งปฏิบัติการ (Directives)]
                        ห้ามเกริ่นนำ ห้ามมีคำลงท้าย ใช้ศัพท์ Quant สถาบัน (กระชับ/ดุดัน) วิเคราะห์ 3 ข้อ:
                        
                        1. ⚔️ พลังโจมตี (Aggressive Exposure): 
                           - โพยชุดนี้ "บุกหนัก" สมใจนโยบายพอร์ตหรือไม่? 
                           - ถ้า VIX ต่ำ (ตลาดกระทิง) แล้วระบบดันสั่งซื้อหุ้น Defensive (เช่น KO, JNJ) ให้ด่าระบบเลยว่าปอดแหกเกินไป!
                        2. 🚨 จุดตายที่แท้จริง (Fatal Flaw): 
                           - กองทุนเราไม่กลัวราคาเหวี่ยง (Volatility) ดังนั้นไม่ต้องเตือนเรื่องนี้! 
                           - ให้หา "จุดตายระดับโครงสร้าง" ที่จะทำให้หุ้นที่สั่งซื้อ "เจ๊งหรือเสียเปรียบเชิงแข่งขันถาวร" แทน
                        3. 🏁 CRO Verdict (คำตัดสินขั้นเด็ดขาด): 
                           - ฟันธง 1 อย่าง: "🟢 APPROVED (ทะลวงบุก!)", "🟡 MODIFY (ลดกองหลัง เพิ่มกองหน้าตัวไหน?)", หรือ "🔴 VETO (ตลาดแพนิกหนัก ถือเงินสดรอช้อน)"
                           - อธิบายเหตุผลแบบ Quant สายบุก สั้นๆ 1 ประโยค
                        """
                        
                        response = model.generate_content(prompt)
                        st.info(response.text)
                        
                    except Exception as e:
                        st.error(f"❌ ระบบ AI ขัดข้อง: {e}")
