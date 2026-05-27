import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json
import warnings

# ==========================================
# 🛠️ ทะลวงกำแพงโฟลเดอร์ (Bulletproof Path Resolver)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ==========================================
# 📥 นำเข้าสมองกลจากไฟล์ต่างๆ (Modular Imports)
# ==========================================
try:
    from data_loader import fetch_fundamental_data, fetch_market_data
    from factors import calc_zscore, calculate_rsi, check_doi_risk, find_sr_levels
    from optimizer import run_black_litterman
    from risk import run_institutional_audit
except ModuleNotFoundError as e:
    st.error(f"🚨 **ระบบหาไฟล์ไม่เจอ!** \n\n{e}\n\n**วิธีแก้:** โปรดตรวจสอบบน GitHub ว่าไฟล์ `data_loader.py`, `factors.py`, `optimizer.py`, `risk.py` ถูกสร้างไว้ที่ **หน้าแรกสุด** ของโปรเจกต์")
    st.stop()

warnings.filterwarnings("ignore")
PORTFOLIO_FILE = "my_portfolio_data.csv"

# ==========================================
# 💾 ฐานข้อมูลถอดรหัสและ Thesis Layer 
# ==========================================
SECTOR_DB = {
    "💻 Tech": ["NVDA", "MSFT", "GOOG", "META", "CSCO", "TXN", "AAPL", "AMD", "PLTR", "AVGO", "RKLB"],
    "🛍️ Consumer": ["COST", "KO", "PEP", "BLK", "MELI", "V", "MA", "WMT"],
    "🏦 Finance": ["BRK-B", "JPM"],
    "🩺 Health": ["UNH", "JNJ", "ISRG", "LLY", "ABBV"],
    "🚗 EV": ["TSLA", "UBER", "TM"],
    "🌿 Green": ["FSLR", "ENPH", "NEE", "SEDG"]
}
ticker_to_sector = {ticker: sector for sector, tickers in SECTOR_DB.items() for ticker in tickers}
THESIS_DB = {"NVDA": "AI Infra Dominance", "JNJ": "Healthcare Titan", "TSLA": "EV & AI Robotics", "ENPH": "Solar Inverter Leader", "TXN": "Analog Chip Moat", "BRK-B": "Fortress Balance Sheet", "RKLB": "Space Infrastructure"} 

# 💎 ฝังพอร์ตจริงเป็นค่าเริ่มต้น (แก้ปัญหาคลาวด์ลบความจำ)
DEFAULT_PORTFOLIO = {
    "BRK-B": 1361.56,
    "NVDA": 1128.72,
    "JNJ": 745.42,
    "TSLA": 475.33,
    "ENPH": 311.58,
    "TXN": 308.51,
    "RKLB": 156.75
}

st.set_page_config(page_title="QuantHQ DCA", page_icon="🛡️", layout="wide")
st.title("🛡️ QUANT-HQ DCA (V. Modular OS)")

# ================= UI & SIDEBAR =================
default_tickers_str = ", ".join(DEFAULT_PORTFOLIO.keys())

if os.path.exists(PORTFOLIO_FILE):
    try:
        temp_df = pd.read_csv(PORTFOLIO_FILE)
        if "รายชื่อหุ้น" in temp_df.columns:
            valid_tickers = [str(t) for t in temp_df["รายชื่อหุ้น"].tolist() if str(t).strip()]
            if valid_tickers: default_tickers_str = ", ".join(valid_tickers)
    except: pass

st.sidebar.subheader("🗂️ หุ้นในพอร์ตของคุณ")
tickers_input = st.sidebar.text_area("รายชื่อหุ้น (คั่นด้วยลูกน้ำ)", default_tickers_str)
my_portfolio = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

engine_choice = st.sidebar.radio("เข็มทิศการลงทุน:", ["🧠 Auto-Pilot (BL + Adaptive)", "🛡️ Safe Mode (Risk Parity)"])
base_lambda = st.sidebar.slider("ระดับความกลัวตั้งต้น", 0.1, 10.0, 2.0, 0.1)

col_input1, col_input2 = st.columns(2)
with col_input1: actual_budget = st.number_input("💵 งบประมาณจัดสรรรอบนี้ (บาท)", 100, 50000, 500, 100) # ตั้งเป้าที่ 500 ตามวินัยกัปตัน
with col_input2: min_order_thb = st.number_input("🚦 ยอดซื้อขั้นต่ำต่อหุ้น (บาท)", 10, 500, 50, 10)

saved_dict = {}
if os.path.exists(PORTFOLIO_FILE):
    try:
        saved_df = pd.read_csv(PORTFOLIO_FILE)
        saved_dict = dict(zip(saved_df["รายชื่อหุ้น"], pd.to_numeric(saved_df["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0.0)))
    except: pass

# โหลดค่าจากพอร์ตเริ่มต้นที่ฝังไว้ ถ้าระบบจำไม่ได้
default_rows = []
for t in my_portfolio:
    if t in saved_dict: val = saved_dict[t]
    else: val = DEFAULT_PORTFOLIO.get(t, 0.0)
    default_rows.append({"รายชื่อหุ้น": t, "ยอดเงินปัจจุบัน (บาท)": val})

df_holdings_edited = st.data_editor(pd.DataFrame(default_rows), use_container_width=True, hide_index=True)
df_holdings_edited.to_csv(PORTFOLIO_FILE, index=False)
current_thb = dict(zip(df_holdings_edited["รายชื่อหุ้น"], df_holdings_edited["ยอดเงินปัจจุบัน (บาท)"]))

# ================= EXECUTION & STATE MANAGEMENT =================
if st.button("🚀 รันระบบ Quant Matrix", type="primary"):
    st.session_state['run_quant_engine'] = True
    st.session_state['matrix_calculated'] = False 

if st.session_state.get('run_quant_engine', False):
    
    # --- PHASE 1: การคำนวณ (ทำแค่รอบเดียว) ---
    if not st.session_state.get('matrix_calculated', False):
        status_box = st.status("🔮 เดินเครื่องสมองกลประมวลผล Matrix สากล...", expanded=True)
        
        benchmark, vix_ticker = 'VOO', '^VIX'
        market_data = fetch_market_data([benchmark, vix_ticker], period="3y")
        vix_current = market_data[vix_ticker].iloc[-1] if vix_ticker in market_data else 20.0
        
        max_sector_cap = 0.40
        turnover_penalty = 0.02
        w_mom, w_quality, w_value = 0.4, 0.3, 0.3
        is_panic = False
        
        if vix_current > 25:
            is_panic, turnover_penalty, max_sector_cap, w_mom, w_quality, w_value = True, 0.05, 0.30, 0.2, 0.5, 0.3
        elif vix_current < 15:
            turnover_penalty, max_sector_cap, w_mom, w_quality, w_value = 0.01, 0.50, 0.5, 0.3, 0.2
        
        voo_ret = market_data[benchmark].pct_change().dropna()
        dynamic_lambda = base_lambda * (voo_ret.tail(30).std() / voo_ret.tail(252).std() if voo_ret.tail(252).std() > 0 else 1.0)
        
        market_proxy = [t for sublist in SECTOR_DB.values() for t in sublist]
        universe_list = list(set(my_portfolio + market_proxy))
        
        prices_3y = fetch_market_data(universe_list, period="3y")
        prices_1y = prices_3y.tail(252)
        returns_1y = prices_1y.pct_change().dropna()
        max_dd = ((prices_1y - prices_1y.cummax()) / prices_1y.cummax()).min() * 100
        
        df_fundamentals = fetch_fundamental_data(prices_1y.columns.tolist())
        metrics, rsi_data, sr_data = [], {}, {}
        mkt_ret_aligned = voo_ret.reindex(returns_1y.index).fillna(0)
        
        for t in prices_1y.columns:
            try:
                cov = np.cov(returns_1y[t], mkt_ret_aligned)[0][1]
                var = np.var(mkt_ret_aligned)
                beta = cov / var if var > 0 else 1.0
                
                s_t, s_m = prices_1y[t].dropna(), market_data[benchmark].dropna()
                ret_6m = (s_t.iloc[-1]/s_t.iloc[-126]) - 1 if len(s_t)>=126 else (s_t.iloc[-1]/s_t.iloc[0]) - 1
                mkt_ret_6m = (s_m.iloc[-1]/s_m.iloc[-126]) - 1 if len(s_m)>=126 else (s_m.iloc[-1]/s_m.iloc[0]) - 1
                residual_mom = ret_6m - (beta * mkt_ret_6m)
            except: residual_mom = 0.0
            
            metrics.append({'Ticker': t, 'Residual_Mom': residual_mom})
            rsi_data[t] = calculate_rsi(prices_1y[t].dropna()).iloc[-1] if len(prices_1y[t].dropna()) > 14 else 50.0
            sr_data[t] = find_sr_levels(prices_1y[t].dropna())
            
        df_metrics = pd.merge(pd.DataFrame(metrics), df_fundamentals, on='Ticker')
        z_mom = calc_zscore(df_metrics['Residual_Mom']).fillna(0)
        z_quality = (calc_zscore(df_metrics['ROA']).fillna(0) + calc_zscore(df_metrics['Margin']).fillna(0)) / 2
        z_value = (calc_zscore(df_metrics['FCF_Yield']).fillna(0) + (calc_zscore(df_metrics['PEG']).fillna(0) * -1)) / 2
        
        df_metrics['Alpha_Score'] = ((z_mom * w_mom) + (z_quality * w_quality) + (z_value * w_value)).fillna(0)
        df_metrics['Max_Drawdown'] = df_metrics['Ticker'].map(max_dd)
        
        final_df = df_metrics.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        port_df['Sector'] = port_df['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
        current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(len(port_df))

        if "Auto-Pilot" in engine_choice:
            try:
                opt_result = run_black_litterman(port_df, returns_1y, dynamic_lambda, turnover_penalty, max_sector_cap, current_weights_arr)
                if opt_result.success: port_df['Target_%'] = port_df['Ticker'].map(dict(zip(port_df['Ticker'].tolist(), opt_result.x))) * 100.0
                else: raise ValueError("Matrix Non-Convergence")
            except:
                port_df['Target_%'] = (1.0 / returns_1y.std()) / (1.0 / returns_1y.std()).sum() * 100.0
        else:
            port_df['Target_%'] = (1.0 / returns_1y.std()) / (1.0 / returns_1y.std()).sum() * 100.0

        for _, row in port_df.groupby('Sector')['Target_%'].sum().reset_index().iterrows():
            if row['Target_%'] > max_sector_cap * 100:
                port_df.loc[port_df['Sector'] == row['Sector'], 'Target_%'] *= (max_sector_cap * 100) / row['Target_%']
        if port_df['Target_%'].sum() > 0: port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0

        fomo_list = []
        for t in port_df['Ticker']:
            if rsi_data.get(t, 50) > 75:
                penalty = max(0.2, 1.0 - ((rsi_data.get(t) - 75) / 25))
                port_df.loc[port_df['Ticker'] == t, 'Target_%'] *= penalty
                fomo_list.append(f"{t} (-{(1-penalty)*100:.0f}%)")
        if fomo_list and port_df['Target_%'].sum() > 0: port_df['Target_%'] = (port_df['Target_%'] / port_df['Target_%'].sum()) * 100.0

        port_df['Target_Val'] = (total_port_value + actual_budget) * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current']
        
        port_df['Buy_Amount'], port_df['Sell_Amount'] = 0.0, 0.0
        buy_mask = port_df['Deficit'] > min_order_thb
        sum_def = port_df.loc[buy_mask, 'Deficit'].sum()
        if sum_def > 0: port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Deficit'] / sum_def) * actual_budget
        
        sell_mask = (port_df['Deficit'] < -min_order_thb) & (port_df['Weight_%'] > (max_sector_cap*100)/2)
        port_df.loc[sell_mask, 'Sell_Amount'] = port_df.loc[sell_mask, 'Deficit'].abs().round(2)
        
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ', 'Sell_Amount': 'ขาย'})
        out['Thesis'] = out['หุ้น'].map(lambda x: THESIS_DB.get(x, "Quant Alpha"))
        out['MDD'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk)
        out['รับ/ต้าน'] = out['หุ้น'].map(sr_data)
        
        top_alpha_display = final_df.head(10).copy() 
        top_alpha_display['Sector'] = top_alpha_display['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        top_alpha_display['Alpha_Score'] = top_alpha_display['Alpha_Score'].round(2)
        top_alpha_display['MDD'] = top_alpha_display['Max_Drawdown'].round(1).astype(str) + "%"
        top_alpha_display['สถานะ'] = top_alpha_display['Ticker'].apply(lambda x: "💼 ถืออยู่" if x in my_portfolio else "✨ เป้าหมายใหม่")

        t_exposure = [{"Sector": r['Sector'], "Weight_%": round(r['Target_%'], 1)} for _, r in port_df.groupby('Sector')['Target_%'].sum().reset_index().iterrows()]
        p_state = json.dumps({"market_regime": "PANIC" if is_panic else "NORMAL", "proposed_buys": out[out['ซื้อ']>0][['หุ้น', 'ซื้อ']].to_dict('records'), "target_sector_exposure": t_exposure, "top_alpha": final_df.head(5)[['Ticker', 'Alpha_Score']].to_dict('records')})

        # 🧠 บันทึก state
        st.session_state['vix_current'] = vix_current
        st.session_state['is_panic'] = is_panic
        st.session_state['fomo_list'] = fomo_list
        st.session_state['out_table'] = out
        st.session_state['top_alpha_table'] = top_alpha_display
        st.session_state['p_state_json'] = p_state
        st.session_state['target_sector_exposure'] = t_exposure
        st.session_state['max_sector_cap_val'] = max_sector_cap
        
        status_box.update(label="--- Quant Engine ประมวลผลเสร็จสิ้น ---", state="complete")
        st.session_state['matrix_calculated'] = True
        
    # --- PHASE 2: ดึงข้อมูลจากความจำ ---
    if st.session_state['is_panic']:
        st.error(f"🚨 Regime: PANIC (VIX = {st.session_state['vix_current']:.1f}) ลด Momentum, เพิ่ม Quality, บังคับ Sector < 30%")
    elif st.session_state['vix_current'] < 15:
        st.success(f"🐂 Regime: BULL (VIX = {st.session_state['vix_current']:.1f}) เร่งเครื่อง Momentum, ขยาย Sector Limit เป็น 50%")
    
    st.markdown("### 📋 2. ตาราง Quant Allocation")
    if st.session_state['fomo_list']: st.warning(f"📉 ตรวจพบหุ้น Overbought ทำการลดเป้าหมาย: {', '.join(st.session_state['fomo_list'])}")
    st.dataframe(st.session_state['out_table'][['หุ้น', 'Thesis', 'MDD', 'RSI', 'รับ/ต้าน', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']].round(2).sort_values('ซื้อ', ascending=False), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🏆 📡 [RADAR] TOP ALPHA CANDIDATES")
    st.dataframe(st.session_state['top_alpha_table'][['Ticker', 'Sector', 'Alpha_Score', 'MDD', 'สถานะ']].rename(columns={'Ticker': 'หุ้น', 'Alpha_Score': 'Alpha Score', 'MDD': 'Max Drawdown'}), use_container_width=True, hide_index=True)

    # --- PHASE 3: AI Audit ---
    if st.button("🧠 รันการตรวจสอบ (Run AI Audit)", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key: st.error("❌ ไม่พบ API Key! โปรดใส่ GEMINI_API_KEY ใน Settings > Secrets")
        else:
            with st.spinner("Analyzing Institutional Risk..."):
                cro_data, pm_data = run_institutional_audit(api_key, st.session_state['p_state_json'])
                col1, col2 = st.columns(2)
                with col1: st.json(cro_data)
                with col2: st.json(pm_data)
                
                conf = pm_data.get("alpha_alignment_score", 0.0)
                t_exposure = st.session_state['target_sector_exposure']
                max_cap = st.session_state['max_sector_cap_val']
                
                if any(sec['Weight_%'] > (max_cap*100) + 0.5 for sec in t_exposure): 
                    st.error(f"🔴 BLOCKED: {cro_data.get('audit_explanation')}")
                elif conf > 0.5: 
                    st.success(f"🟢 APPROVED: {pm_data.get('audit_explanation')}")
                else: 
                    st.warning(f"🟡 WARNING: {pm_data.get('audit_explanation')}")
