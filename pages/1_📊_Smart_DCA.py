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
st.title("🛡️ QUANT-HQ DCA (Phase B: Advanced Execution)")

# --- Session State สำหรับนับรอบ Confirmation (กัน Whipsaw) ---
if 'entry_candidates_history' not in st.session_state: st.session_state['entry_candidates_history'] = {}
if 'rotation_history' not in st.session_state: st.session_state['rotation_history'] = {}

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
        
        status_box.update(label="📡 Layer 3: ดึงข้อมูล Macro และคำนวณ HMM...")
        df_macro = regime_engine.fetch_macro_features()
        raw_probs = regime_engine.expanding_fit_predict(df_macro)
        smooth_probs = regime_engine.apply_transition_smoothing(raw_probs)
        hysteresis_probs = regime_engine.apply_hysteresis(smooth_probs)
        regime_weights = factor_allocator.calculate_weights(hysteresis_probs)
        P_PANIC = regime_weights.get('P_PANIC', 0.0)
        
        status_box.update(label="🧼 Layer 1: ร่อนหุ้นผ่าน Sanity Check Pipeline...")
        clean_universe = pipeline.filter_universe()
        final_scan_list = list(set(my_portfolio + clean_universe))
        
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
            except: 
                residual_mom = 0.0
                beta = 1.0
            
            # 🛠️ บันทึกค่า Beta เพื่อนำไปใช้ใน Decision 1 และ 3
            metrics.append({'Ticker': t, 'Residual_Mom': residual_mom, 'Beta': beta})
            
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

        status_box.update(label="🏛️ Layer 4: Optimizer & Execution Gate...")
        if "Auto-Pilot" in engine_choice:
            opt_res, mu_bl = run_institutional_black_litterman(
                port_df=port_df, returns_df=returns_1y, regime_probs=regime_weights, current_weights=current_weights_arr, risk_aversion=base_lambda
            )
            if opt_res.success:
                port_df['Target_%'] = port_df['Ticker'].map(dict(zip(port_df['Ticker'].tolist(), opt_res.x))) * 100.0
            else:
                rp_w = 1.0 / (returns_1y[port_df['Ticker'].tolist()].std() + 1e-9)
                port_df['Target_%'] = port_df['Ticker'].map(((rp_w / rp_w.sum()) * 100.0).to_dict())
        else:
            rp_w = 1.0 / (returns_1y[port_df['Ticker'].tolist()].std() + 1e-9)
            port_df['Target_%'] = port_df['Ticker'].map(((rp_w / rp_w.sum()) * 100.0).to_dict())

        # Threshold Rebalancing (กัน Trade บ่อยเกินไป)
        MIN_DEVIATION = 5.0  
        mask_small_diff = (port_df['Target_%'] - port_df['Weight_%']).abs() < MIN_DEVIATION
        port_df.loc[mask_small_diff, 'Target_%'] = port_df.loc[mask_small_diff, 'Weight_%']
        
        total_target = port_df['Target_%'].sum()
        if total_target > 0: port_df['Target_%'] = (port_df['Target_%'] / total_target) * 100.0

        port_df['Target_Val'] = (total_port_value + actual_budget) * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current']
        port_df['Buy_Amount'], port_df['Sell_Amount'] = 0.0, 0.0

        # ==========================================
        # 🟢 DECISION 1: DCA Allocation Gate
        # ==========================================
        port_df['Beta'] = port_df['Ticker'].map(dict(zip(df_metrics['Ticker'], df_metrics['Beta'])))
        port_df['Regime_Weight'] = 1.0 - (P_PANIC * port_df['Beta'].clip(0, 2) / 2)
        port_df['Weighted_Deficit'] = port_df['Deficit'] * port_df['Regime_Weight']
        
        # กรอง Alpha > 0 ถึงจะซื้อเพิ่ม
        buy_mask = (port_df['Weighted_Deficit'] > min_order_thb) & (port_df['Alpha_Score'] > 0)
        sum_def = port_df.loc[buy_mask, 'Weighted_Deficit'].sum()
        if sum_def > 0: 
            port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Weighted_Deficit'] / sum_def) * actual_budget
        
        # จัดตารางเตรียมแสดงผล
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

        # ==========================================
        # 🚨 DECISION 3: Exit Rules
        # ==========================================
        def evaluate_exit_signals(df, p_panic):
            exit_signals = []
            for _, row in df.iterrows():
                t = row['หุ้น']
                reasons = []
                severity = 'HOLD'
                
                if row['Alpha_Score'] < -0.5:
                    reasons.append(f"Alpha ติดลบ ({row['Alpha_Score']:.2f})")
                    severity = 'REDUCE'
                    
                dd_threshold = -35.0 if row['Max_Drawdown'] < -30.0 else -25.0
                if row['Max_Drawdown'] < dd_threshold:
                    reasons.append(f"Drawdown เกิน threshold ({row['Max_Drawdown']:.1f}%)")
                    severity = 'EXIT'
                    
                beta = row.get('Beta', 1.0)
                if p_panic > 0.7 and beta > 1.5 and row['Alpha_Score'] < 0:
                    reasons.append(f"PANIC + High Beta ({beta:.1f}) + Alpha ติดลบ")
                    severity = 'REDUCE'
                    
                if reasons:
                    exit_signals.append({'Ticker': t, 'Severity': severity, 'Reasons': reasons})
            return exit_signals
        
        st.session_state['exit_signals'] = evaluate_exit_signals(out, P_PANIC)

        # ==========================================
        # 🌟 DECISION 2: New Position Logic
        # ==========================================
        def evaluate_new_entries(candidates_df, port_df, p_panic, my_portfolio, max_stocks=10):
            if len(my_portfolio) >= max_stocks: return []
            port_avg_alpha = port_df['Alpha_Score'].mean() if not port_df.empty else 0
            worst_risk_adj = (port_df['Alpha_Score'] / (port_df['Max_Drawdown'].abs() + 1e-9)).min() if not port_df.empty else -999
            
            candidates = candidates_df[candidates_df['สถานะ'] == "✨ เป้าหมายใหม่"]
            passed_tickers = []
            for _, row in candidates.iterrows():
                if (row['Alpha_Score'] > port_avg_alpha + 1.0) and \
                   (row['Risk_Adj_Alpha'] > worst_risk_adj) and \
                   not (p_panic > 0.5 and float(row['MDD'].replace('%','')) < -30):
                    passed_tickers.append(row['Ticker'])
            return passed_tickers

        new_entries = evaluate_new_entries(top_alpha_display, port_df, P_PANIC, my_portfolio)
        for t in new_entries:
            st.session_state['entry_candidates_history'][t] = st.session_state['entry_candidates_history'].get(t, 0) + 1
        for t in list(st.session_state['entry_candidates_history'].keys()):
            if t not in new_entries: st.session_state['entry_candidates_history'][t] = 0
            
        st.session_state['confirmed_new_entries'] = [t for t, c in st.session_state['entry_candidates_history'].items() if c >= 2]

        # ==========================================
        # 🔄 DECISION 4: Rotation Logic
        # ==========================================
        def evaluate_rotation(top_alpha_display, p_panic, actual_budget):
            current_port = top_alpha_display[top_alpha_display['สถานะ'] == "💼 ถืออยู่"]
            outside = top_alpha_display[top_alpha_display['สถานะ'] == "✨ เป้าหมายใหม่"]
            if current_port.empty or outside.empty: return None
            
            worst_held = current_port.sort_values('Risk_Adj_Alpha').iloc[0]
            best_new = outside.sort_values('Risk_Adj_Alpha', ascending=False).iloc[0]
            
            alpha_diff = best_new['Alpha_Score'] - worst_held['Alpha_Score']
            risk_adj_diff = best_new['Risk_Adj_Alpha'] - worst_held['Risk_Adj_Alpha']
            
            if alpha_diff < 1.0 or risk_adj_diff < 0.005: return None
            if p_panic > 0.6: return None
            
            estimated_pos = actual_budget * 10
            rotation_cost = estimated_pos * 0.001 * 2
            estimated_gain = alpha_diff * 0.02 * estimated_pos
            if estimated_gain < rotation_cost * 3: return None
            
            return {'buy': best_new['Ticker'], 'sell': worst_held['Ticker'], 'alpha_diff': alpha_diff}

        rot_signal = evaluate_rotation(top_alpha_display, P_PANIC, actual_budget)
        if rot_signal:
            key = f"{rot_signal['buy']}_vs_{rot_signal['sell']}"
            st.session_state['rotation_history'][key] = st.session_state['rotation_history'].get(key, 0) + 1
            if st.session_state['rotation_history'][key] >= 2:
                rot_signal['status'] = 'CONFIRMED'
            else:
                rot_signal['status'] = 'PENDING'
            st.session_state['rotation_alert'] = rot_signal
        else:
            st.session_state['rotation_alert'] = None

        t_exposure = [{"Sector": r['Sector'], "Weight_%": round(r['Target_%'], 1)} for _, r in port_df.groupby('Sector')['Target_%'].sum().reset_index().iterrows()]
        p_state = json.dumps({"market_regime": f"{regime_weights['Current_State']} | P(Bull)={regime_weights['P_BULL']*100:.0f}%", "proposed_buys": out[out['ซื้อ']>0][['หุ้น', 'ซื้อ']].to_dict('records')})

        st.session_state['regime_report'] = f"📊 HMM Regime -> State: {regime_weights['Current_State']} | 🐂 P(Bull): {regime_weights['P_BULL']*100:.1f}% | 🐻 P(Panic): {regime_weights['P_PANIC']*100:.1f}%"
        st.session_state['factor_mix'] = f"⚡ Dynamic Factors -> Mom: {w_m*100:.0f}% | Qual: {w_q*100:.0f}% | Val: {w_v*100:.0f}%"
        st.session_state['out_table'] = out
        st.session_state['top_alpha_table'] = top_alpha_display
        st.session_state['p_state_json'] = p_state
        st.session_state['target_sector_exposure'] = t_exposure
        
        status_box.update(label="--- 4 Layers Framework ประมวลผลเสร็จสิ้น ---", state="complete")
        st.session_state['matrix_calculated'] = True
        
    st.info(st.session_state['regime_report'])
    st.success(st.session_state['factor_mix'])
    
    # --- แสดงผล Alerts (Decision 2, 3, 4) ---
    if st.session_state.get('exit_signals'):
        st.markdown("### 🚨 ระบบป้องกันวินาศภัย (EXIT SIGNALS)")
        for sig in st.session_state['exit_signals']:
            if sig['Severity'] == 'EXIT':
                st.error(f"**🔴 EXIT (พิจารณาขายทิ้ง):** {sig['Ticker']} — {' | '.join(sig['Reasons'])}")
            elif sig['Severity'] == 'REDUCE':
                st.warning(f"**🟡 REDUCE (หยุดซื้อ/ลดน้ำหนัก):** {sig['Ticker']} — {' | '.join(sig['Reasons'])}")

    if st.session_state.get('confirmed_new_entries'):
        st.markdown("### 🌟 แจ้งเตือนหุ้นใหม่ (NEW POSITIONS)")
        for t in st.session_state['confirmed_new_entries']:
            st.success(f"**🟢 CONFIRMED:** {t} ผ่านเกณฑ์ 3 ข้อติดต่อกัน 2 รอบแล้ว พิจารณาเพิ่มเข้าพอร์ต!")

    if st.session_state.get('rotation_alert'):
        rot = st.session_state['rotation_alert']
        st.markdown("### 🔄 ระบบสับเปลี่ยนหุ้น (ROTATION)")
        if rot['status'] == 'CONFIRMED':
            st.success(f"**✅ CONFIRMED ROTATION:** แนะนำให้ขาย **{rot['sell']}** แล้วย้ายเข้า **{rot['buy']}** (ความต่าง Alpha: {rot['alpha_diff']:.2f})")
        else:
            st.info(f"**⏳ PENDING ROTATION:** เล็งเห็นโอกาสขาย **{rot['sell']}** เข้า **{rot['buy']}** (รอการยืนยันในรอบถัดไป)")

    st.markdown("### 📋 2. ตาราง Quant Allocation (Alpha Gate & Threshold Adjusted)")
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
