import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json
import warnings

# ทะลวงโฟลเดอร์
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from data_pipeline import InstitutionalDataPipeline
    from factors import two_pass_zscore, calculate_alpha_decay, calculate_rsi, check_doi_risk, find_sr_levels
    from regime import MarketRegimeHMM, DynamicFactorAllocator
    from optimizer import run_institutional_black_litterman
    from risk import run_institutional_audit
except ModuleNotFoundError as e:
    st.error(f"🚨 **ระบบหาไฟล์ไม่เจอ!** \n\n{e}")
    st.stop()

warnings.filterwarnings("ignore")
PORTFOLIO_FILE = "my_portfolio_data.csv"

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

DEFAULT_PORTFOLIO = {"BRK-B": 1361.56, "NVDA": 1128.72, "JNJ": 745.42, "TSLA": 475.33, "ENPH": 311.58, "TXN": 308.51, "RKLB": 156.75}

st.set_page_config(page_title="QuantHQ DCA", page_icon="🛡️", layout="wide")
st.title("🛡️ QUANT-HQ DCA (Phase A: Institutional Stability)")

st.sidebar.subheader("🗂️ หุ้นในพอร์ตของคุณ")
tickers_input = st.sidebar.text_area("รายชื่อหุ้น (คั่นด้วยลูกน้ำ)", ", ".join(DEFAULT_PORTFOLIO.keys()))
my_portfolio = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

engine_choice = st.sidebar.radio("เข็มทิศการลงทุน:", ["🧠 Auto-Pilot (Dynamic Macro BL)", "🛡️ Safe Mode (Risk Parity)"])
base_lambda = st.sidebar.slider("ระดับความกลัวตั้งต้น", 0.1, 10.0, 2.5, 0.1)

col_input1, col_input2 = st.columns(2)
with col_input1: actual_budget = st.number_input("💵 งบประมาณจัดสรรรอบนี้ (บาท)", 100, 50000, 500, 100)
with col_input2: min_order_thb = st.number_input("🚦 ยอดซื้อขั้นต่ำต่อหุ้น (บาท)", 10, 500, 50, 10)

saved_dict = {}
if os.path.exists(PORTFOLIO_FILE):
    try:
        saved_df = pd.read_csv(PORTFOLIO_FILE)
        saved_dict = dict(zip(saved_df["รายชื่อหุ้น"], pd.to_numeric(saved_df["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').fillna(0.0)))
    except: pass

default_rows = []
for t in my_portfolio:
    val = saved_dict.get(t, DEFAULT_PORTFOLIO.get(t, 0.0))
    default_rows.append({"รายชื่อหุ้น": t, "ยอดเงินปัจจุบัน (บาท)": val})

df_holdings_edited = st.data_editor(pd.DataFrame(default_rows), use_container_width=True, hide_index=True)
df_holdings_edited.to_csv(PORTFOLIO_FILE, index=False)
current_thb = dict(zip(df_holdings_edited["รายชื่อหุ้น"], df_holdings_edited["ยอดเงินปัจจุบัน (บาท)"]))

if st.button("🚀 รันระบบ Quant Matrix", type="primary"):
    st.session_state['run_quant_engine'] = True
    st.session_state['matrix_calculated'] = False 

if st.session_state.get('run_quant_engine', False):
    
    if not st.session_state.get('matrix_calculated', False):
        status_box = st.status("🔮 กำลังเดินเครื่องสถาปัตยกรรม 4 Layers...", expanded=True)
        
        market_proxy = [t for sublist in SECTOR_DB.values() for t in sublist]
        total_universe = list(set(my_portfolio + market_proxy))
        
        pipeline = InstitutionalDataPipeline(total_universe)
        regime_engine = MarketRegimeHMM(n_states=2)
        factor_allocator = DynamicFactorAllocator()
        
        status_box.update(label="📡 Layer 3: ดึงข้อมูล Macro และคำนวณ HMM (Hysteresis Active)...")
        df_macro = regime_engine.fetch_macro_features()
        raw_probs = regime_engine.expanding_fit_predict(df_macro)
        smooth_probs = regime_engine.apply_transition_smoothing(raw_probs)
        # เรียกใช้ Hysteresis
        hysteresis_probs = regime_engine.apply_hysteresis(smooth_probs)
        regime_weights = factor_allocator.calculate_weights(hysteresis_probs)
        
        status_box.update(label="🧼 Layer 1: ร่อนหุ้นผ่าน Sanity Check Pipeline...")
        clean_universe = pipeline.filter_universe()
        final_scan_list = list(set(my_portfolio + clean_universe))
        
        # ดึงราคาผ่านระบบ Validate
        raw_prices = pipeline.fetch_bulk_market_data(final_scan_list)
        prices_df = raw_prices.copy()
        prices_1y = prices_df.tail(252)
        
        returns_1y = prices_1y.pct_change().fillna(0)
        max_dd = ((prices_1y - prices_1y.cummax()) / prices_1y.cummax()).min() * 100
        
        status_box.update(label="🧬 Layer 2: สกัด Factor และคำนวณ Z-Score...")
        from data_loader import fetch_fundamental_data
        df_fundamentals = fetch_fundamental_data(final_scan_list)
        df_clean_fundamentals = pipeline.clean_fundamentals(df_fundamentals)
        
        metrics = []
        rsi_data, sr_data = {}, {}
        spy_ret = df_macro['SPY_Ret'].tail(252).reindex(returns_1y.index).fillna(0)
        
        for t in final_scan_list:
            try:
                cov = returns_1y[t].cov(spy_ret)
                var = spy_ret.var()
                beta = cov / var if var > 0 else 1.0
                
                s_t = prices_1y[t].dropna()
                ret_6m = (s_t.iloc[-1]/s_t.iloc[max(0, len(s_t)-126)]) - 1 if len(s_t)>20 else 0.0
                mkt_ret_6m = (1 + spy_ret.tail(126)).prod() - 1 
                residual_mom = ret_6m - (beta * mkt_ret_6m)
            except: residual_mom = 0.0
            
            metrics.append({'Ticker': t, 'Residual_Mom': residual_mom})
            
            clean_series = prices_1y[t].dropna()
            rsi_data[t] = calculate_rsi(clean_series).iloc[-1] if len(clean_series) > 14 else 50.0
            sr_data[t] = find_sr_levels(clean_series)
            
        df_metrics = pd.merge(pd.DataFrame(metrics), df_clean_fundamentals, on='Ticker').fillna(0.0)
        df_metrics['Sector'] = df_metrics['Ticker'].map(lambda x: ticker_to_sector.get(x, '🧩 Others'))
        
        col_roa = 'ROA_PIT' if 'ROA_PIT' in df_metrics.columns else 'ROA'
        col_margin = 'Margin_PIT' if 'Margin_PIT' in df_metrics.columns else 'Margin'
        col_fcf_yield = 'FCF_Yield_PIT' if 'FCF_Yield_PIT' in df_metrics.columns else 'FCF_Yield'
        col_peg = 'PEG_PIT' if 'PEG_PIT' in df_metrics.columns else 'PEG'

        z_mom = two_pass_zscore(df_metrics, 'Residual_Mom', 'Sector')
        z_quality = (two_pass_zscore(df_metrics, col_roa, 'Sector') + two_pass_zscore(df_metrics, col_margin, 'Sector')) / 2
        z_value = (two_pass_zscore(df_metrics, col_fcf_yield, 'Sector') + (two_pass_zscore(df_metrics, col_peg, 'Sector') * -1)) / 2
        
        w_m, w_q, w_v = regime_weights['Mom'], regime_weights['Qual'], regime_weights['Val']
        df_metrics['Alpha_Score'] = (z_mom * w_m) + (z_quality * w_q) + (z_value * w_v)
        df_metrics['Alpha_Score'] = df_metrics['Alpha_Score'].apply(lambda x: calculate_alpha_decay(x, days_passed=3, half_life_days=30))
        df_metrics['Max_Drawdown'] = df_metrics['Ticker'].map(max_dd)
        
        final_df = df_metrics.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
        current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(len(port_df))

        status_box.update(label="🏛️ Layer 4: Optimizer & Threshold Rebalancing...")
        if "Auto-Pilot" in engine_choice:
            opt_res, mu_bl = run_institutional_black_litterman(
                port_df=port_df,
                returns_df=returns_1y,
                regime_probs=regime_weights,
                current_weights=current_weights_arr,
                risk_aversion=base_lambda
            )
            if opt_res.success:
                port_df['Target_%'] = port_df['Ticker'].map(dict(zip(port_df['Ticker'].tolist(), opt_res.x))) * 100.0
            else:
                rp_w = 1.0 / (returns_1y[port_df['Ticker'].tolist()].std() + 1e-9)
                rp_target = (rp_w / rp_w.sum()) * 100.0
                port_df['Target_%'] = port_df['Ticker'].map(rp_target.to_dict())
        else:
            rp_w = 1.0 / (returns_1y[port_df['Ticker'].tolist()].std() + 1e-9)
            rp_target = (rp_w / rp_w.sum()) * 100.0
            port_df['Target_%'] = port_df['Ticker'].map(rp_target.to_dict())

        # ==========================================
        # 🛡️ PRIORITY 2: THRESHOLD REBALANCING (ลดรอบสับเปลี่ยน)
        # ==========================================
        MIN_DEVIATION = 5.0  
        mask_small_diff = (port_df['Target_%'] - port_df['Weight_%']).abs() < MIN_DEVIATION
        port_df.loc[mask_small_diff, 'Target_%'] = port_df.loc[mask_small_diff, 'Weight_%']
        
        total_target = port_df['Target_%'].sum()
        if total_target > 0: port_df['Target_%'] = (port_df['Target_%'] / total_target) * 100.0

        port_df['Target_Val'] = (total_port_value + actual_budget) * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current']
        
        port_df['Buy_Amount'], port_df['Sell_Amount'] = 0.0, 0.0
        buy_mask = port_df['Deficit'] > min_order_thb
        sum_def = port_df.loc[buy_mask, 'Deficit'].sum()
        if sum_def > 0: port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Deficit'] / sum_def) * actual_budget
        
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ', 'Sell_Amount': 'ขาย'})
        out['Thesis'] = out['หุ้น'].map(lambda x: THESIS_DB.get(x, "Quant Alpha"))
        out['MDD'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk)
        out['รับ/ต้าน'] = out['หุ้น'].map(sr_data)
        
        top_alpha_display = final_df.head(10).copy() 
        top_alpha_display['Risk_Adj_Alpha'] = (top_alpha_display['Alpha_Score'] / (top_alpha_display['Max_Drawdown'].abs() + 1e-9)).round(3)
        top_alpha_display['Alpha_Score'] = top_alpha_display['Alpha_Score'].round(2)
        top_alpha_display['MDD'] = top_alpha_display['Max_Drawdown'].round(1).astype(str) + "%"
        top_alpha_display['สถานะ'] = top_alpha_display['Ticker'].apply(lambda x: "💼 ถืออยู่" if x in my_portfolio else "✨ เป้าหมายใหม่")

        t_exposure = [{"Sector": r['Sector'], "Weight_%": round(r['Target_%'], 1)} for _, r in port_df.groupby('Sector')['Target_%'].sum().reset_index().iterrows()]
        p_state = json.dumps({"market_regime": f"{regime_weights['Current_State']} | P(Bull)={regime_weights['P_BULL']*100:.0f}%", "proposed_buys": out[out['ซื้อ']>0][['หุ้น', 'ซื้อ']].to_dict('records')})

        st.session_state['regime_report'] = f"📊 HMM Regime (Hysteresis Active) -> Current State: {regime_weights['Current_State']} | 🐂 P(Bull): {regime_weights['P_BULL']*100:.1f}% | 🐻 P(Panic): {regime_weights['P_PANIC']*100:.1f}%"
        st.session_state['factor_mix'] = f"⚡ Dynamic Factor Allocation -> 🛠️ Momentum: {w_m*100:.0f}% | 🛡️ Quality: {w_q*100:.0f}% | ⚖️ Value: {w_v*100:.0f}%"
        st.session_state['out_table'] = out
        st.session_state['top_alpha_table'] = top_alpha_display
        st.session_state['p_state_json'] = p_state
        st.session_state['target_sector_exposure'] = t_exposure
        
        status_box.update(label="--- 4 Layers Framework ประมวลผลเสร็จสิ้น ---", state="complete")
        st.session_state['matrix_calculated'] = True
        
    st.info(st.session_state['regime_report'])
    st.success(st.session_state['factor_mix'])
    
    st.markdown("### 📋 2. ตาราง Quant Allocation (Threshold Adjusted)")
    st.dataframe(st.session_state['out_table'][['หุ้น', 'Thesis', 'MDD', 'RSI', 'รับ/ต้าน', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']].round(2).sort_values('ซื้อ', ascending=False), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🏆 📡 [RADAR] TOP ALPHA CANDIDATES (MAD Z-Score)")
    st.dataframe(st.session_state['top_alpha_table'][['Ticker', 'Sector', 'Alpha_Score', 'Risk_Adj_Alpha', 'MDD', 'สถานะ']].rename(columns={'Ticker': 'หุ้น', 'Alpha_Score': 'Alpha Score', 'Risk_Adj_Alpha': 'Risk-adj Alpha', 'MDD': 'Max Drawdown'}), use_container_width=True, hide_index=True)

    if st.button("🧠 รันการตรวจสอบ (Run AI Audit)", type="primary"):
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key: st.error("❌ ไม่พบ API Key! โปรดใส่ GEMINI_API_KEY ใน Settings > Secrets")
        else:
            with st.spinner("Analyzing Institutional Risk..."):
                cro_data, pm_data = run_institutional_audit(api_key, st.session_state['p_state_json'])
                col1, col2 = st.columns(2)
                with col1: st.json(cro_data)
                with col2: st.json(pm_data)
