import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os, sys
import time
import plotly.express as px

warnings.filterwarnings("ignore")

# ==========================================
# 🛠️ ฟังก์ชันคณิตศาสตร์ 
# ==========================================
def calc_zscore(series): return (series - series.mean()) / series.std()
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

def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
def enable_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# ==========================================
# 💾 ฐานข้อมูลความจำ & Thesis Layer (V.50.15)
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
DEFENSIVE_SECTORS = ["🛍️ Consumer", "🩺 Health"]

# 🧠 Thesis Dictionary (อธิบายเหตุผลการลงทุน)
THESIS_DB = {
    "NVDA": "AI Infra Dominance", "MSFT": "Cloud & OS Monopoly",
    "COST": "Membership Cashflow", "AVGO": "Custom Silicon / M&A",
    "V": "Global Toll Network", "KO": "Defensive Cashflow",
    "JNJ": "Healthcare Titan", "RKLB": "Space Infrastructure"
}

if 'dca_budget' not in st.session_state: st.session_state['dca_budget'] = 500.0 
if 'min_order_thb' not in st.session_state: st.session_state['min_order_thb'] = 50.0

st.set_page_config(page_title="QuantHQ DCA", page_icon="🛡️", layout="wide")
st.title("🛡️ QUANT-HQ DCA (V.50.15 Institutional Edition)")
st.markdown("ระบบจัดพอร์ตระดับสถาบัน **(อัปเกรด: VIX Regime, Max Drawdown & Thesis Layer)**")
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
tickers_input = st.sidebar.text_area("รายชื่อหุ้นที่ถืออยู่", default_tickers)
my_portfolio = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.sidebar.markdown("---")
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

saved_dict = {}
if os.path.exists(PORTFOLIO_FILE):
    try:
        saved_df = pd.read_csv(PORTFOLIO_FILE)
        if not saved_df.empty and "รายชื่อหุ้น" in saved_df.columns:
            saved_dict = dict(zip(saved_df["รายชื่อหุ้น"], pd.to_numeric(saved_df["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0)))
    except: pass

default_rows = [{"รายชื่อหุ้น": t, "ยอดเงินปัจจุบัน (บาท)": float(saved_dict.get(t, 0))} for t in my_portfolio]
st.session_state['portfolio_holdings'] = pd.DataFrame(default_rows)
df_holdings_edited = st.data_editor(st.session_state['portfolio_holdings'], use_container_width=True, hide_index=True)
df_holdings_edited.to_csv(PORTFOLIO_FILE, index=False)
current_thb = dict(zip(df_holdings_edited["รายชื่อหุ้น"], pd.to_numeric(df_holdings_edited["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0)))

actual_budget = budget 

if st.button("🚀 รันระบบ Ultimate Rebalancer", type="primary"):
    if not my_portfolio: 
        st.error("⚠️ โปรดระบุชื่อหุ้นก่อนครับ")
    else:
        status_box = st.status(f"🔮 เดินเครื่องสมองกลวิเคราะห์สภาวะตลาด...", expanded=True)
        
        # 🧠 V.50.15: เพิ่ม VIX Index เพื่อจับ Regime ตลาด
        benchmark = 'VOO'
        vix_ticker = '^VIX'
        
        block_print()
        market_data = yf.download([benchmark, vix_ticker], period="3y", progress=False)['Close']
        enable_print()
        
        # จับสภาวะตลาด (Regime Detection)
        vix_current = market_data[vix_ticker].iloc[-1] if vix_ticker in market_data else 20.0
        sma200_voo = market_data[benchmark].rolling(200).mean().iloc[-1]
        is_market_crashing = market_data[benchmark].iloc[-1] < sma200_voo
        
        if vix_current > 25:
            st.error(f"🚨 **Market Regime: PANIC (VIX = {vix_current:.1f})** ตลาดตื่นตระหนก! ระบบลดเพดานหุ้นสายบุก")
            is_panic = True
        elif vix_current < 15:
            st.success(f"🐂 **Market Regime: BULL (VIX = {vix_current:.1f})** ตลาดกระทิง! ระบบเปิดเพดานสายบุกเต็มที่")
            is_panic = False
        else:
            st.info(f"⚖️ **Market Regime: NORMAL (VIX = {vix_current:.1f})** ตลาดปกติ")
            is_panic = False

        market_proxy = [t for sublist in SECTOR_DB.values() for t in sublist]
        universe_list = list(set(my_portfolio + market_proxy)) 
        
        block_print()
        prices_1y = yf.download(universe_list, period="1y", progress=False)['Close']
        enable_print()
        
        returns_1y = prices_1y.pct_change().dropna()
        
        # คำนวณ Max Drawdown (MDD)
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
        
        quality_data, rsi_data = [], {}
        for t in df['Ticker'].tolist():
            roa, margin = None, None
            try:
                block_print()
                info = yf.Ticker(t).info
                enable_print()
                roa, margin = info.get('returnOnAssets'), info.get('profitMargins')
            except: enable_print()
            quality_data.append({'Ticker': t, 'ROA': roa * 100 if roa is not None else np.nan, 'Margin': margin * 100 if margin is not None else np.nan})
            rsi_data[t] = calculate_rsi(prices_1y[t].dropna()).iloc[-1] 
            
        final_df = pd.merge(df, pd.DataFrame(quality_data), on='Ticker')
        final_df['Alpha_Score'] = (calc_zscore(final_df['Raw_Mom']) * 0.5) + (calc_zscore(final_df['ROA'].fillna(final_df['ROA'].median())) * 0.25) + (calc_zscore(final_df['Margin'].fillna(final_df['Margin'].median())) * 0.25)
        
        final_df['Sharpe'] = final_df['Ticker'].map(sharpe_ratio)
        final_df['Sortino'] = final_df['Ticker'].map(sortino_ratio)
        final_df['Max_Drawdown'] = final_df['Ticker'].map(max_dd)
        
        final_df = final_df.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
        
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        port_df['Sector'] = port_df['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        
        total_port_value = port_df['Current'].sum()
        
        # Risk Parity เป็น Core Logic
        port_df['Inv_Vol'] = port_df['Ticker'].map(1.0 / returns_1y.std())
        port_df['Target_%'] = (port_df['Inv_Vol'] / port_df['Inv_Vol'].sum()) * 100.0 if port_df['Inv_Vol'].sum() > 0 else 0.0

        sector_totals = port_df.groupby('Sector')['Current'].sum().reset_index()
        overweight_sectors = []
        
        # 🛡️ อัปเกรด V.50.15: เพดานแปรผันตาม VIX
        max_sector_cap = 50 if is_panic else (55 if is_market_crashing else 70)
        
        for _, row in sector_totals.iterrows():
            sec_weight = (row['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
            if sec_weight > max_sector_cap: overweight_sectors.append(row['Sector'])
            
        if overweight_sectors:
            port_df.loc[port_df['Sector'].isin(overweight_sectors), 'Target_%'] = 0.0
            if port_df['Target_%'].sum() > 0: port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0

        # FOMO Breaker
        fomo_list = [t for t in port_df['Ticker'] if rsi_data.get(t, 50) > 75] 
        if fomo_list:
            port_df.loc[port_df['Ticker'].isin(fomo_list), 'Target_%'] = 0.0
            if port_df['Target_%'].sum() > 0: port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0
            st.warning(f"🛑 **Anti-FOMO:** ระงับซื้อ {', '.join(fomo_list)} กราฟตึงเกินไป")

        # จัดสรรเงิน
        target_total = total_port_value + actual_budget
        port_df['Target_Val'] = target_total * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current'] 
        port_df['Buy_Amount'] = 0.0
        
        buy_mask = port_df['Deficit'] > min_order_thb
        sum_deficit = port_df.loc[buy_mask, 'Deficit'].sum()
        if sum_deficit > 0:
            port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Deficit'] / sum_deficit) * actual_budget
            port_df['Buy_Amount'] = port_df['Buy_Amount'].apply(lambda x: round(x, 2) if x >= min_order_thb else 0.0)
        
        cash_reserve = actual_budget - port_df['Buy_Amount'].sum()
        status_box.update(label="--- คำนวณเสร็จสิ้น! ---", state="complete")
        
        # 🎨 แสดงผลตาราง The Final Auto-Pilot
        st.markdown(f"### 📋 ใบสั่งซื้อ & แดชบอร์ดความเสี่ยง (Thesis Layer)")
        
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Current': 'ทุนเดิม', 'Buy_Amount': 'สั่งซื้อ(บาท)'})
        out['MDD (หลุมยุบ)'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk)
        out['Thesis (ตรรกะลงทุน)'] = out['หุ้น'].map(lambda x: THESIS_DB.get(x, "Quantitative Alpha"))
        
        st.dataframe(out[['หุ้น', 'Thesis (ตรรกะลงทุน)', 'MDD (หลุมยุบ)', 'RSI', 'ทุนเดิม', 'สั่งซื้อ(บาท)']].sort_values('สั่งซื้อ(บาท)', ascending=False), use_container_width=True, hide_index=True)
        
        st.code(f"💰 พักเงินสด = {cash_reserve:.2f} บ.", language="text")
        
        # 🏆 เรดาร์ (เพิ่ม MDD)
        st.markdown("---")
        st.subheader("🏆 📡 [RADAR] TOP 20 ALPHA (รวม Max Drawdown)")
        top20_df = final_df.head(20).copy()
        top20_df['MDD'] = top20_df['Max_Drawdown'].round(1).astype(str) + "%"
        display_top20 = top20_df[['Rank', 'Ticker', 'Alpha_Score', 'Sharpe', 'Sortino', 'MDD']].rename(
            columns={'Ticker': 'หุ้น', 'Alpha_Score': 'Alpha', 'MDD': 'Max Drawdown (ดิ่งลึกสุด)'})
        st.dataframe(display_top20, use_container_width=True, hide_index=True)
