import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json
import warnings
from datetime import datetime
import plotly.express as px

# ==========================================
# 🛠️ ทะลวงกำแพงโฟลเดอร์
# ==========================================
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

# ==========================================
# 💾 ฐานข้อมูลและ Baseline (อัปเดต Bug #3: แยกกลุ่ม BRK-B)
# ==========================================
SECTOR_DB = {
    "💻 Tech": ["NVDA", "MSFT", "GOOG", "META", "CSCO", "TXN", "AAPL", "AMD", "PLTR", "AVGO", "RKLB"],
    "🛍️ Consumer": ["COST", "KO", "PEP", "BLK", "MELI", "V", "MA", "WMT"],
    "🏦 Finance": ["JPM"],
    "🛡️ Diversified (BRK)": ["BRK-B"],
    "🩺 Health": ["UNH", "JNJ", "ISRG", "LLY", "ABBV"],
    "🚗 EV": ["TSLA", "UBER", "TM"],
    "🌿 Green": ["FSLR", "ENPH", "NEE", "SEDG"]
}
ticker_to_sector = {ticker: sector for sector, tickers in SECTOR_DB.items() for ticker in tickers}
THESIS_DB = {"NVDA": "AI Infra Dominance", "JNJ": "Healthcare Titan", "TSLA": "EV & AI Robotics", "ENPH": "Solar Inverter Leader", "TXN": "Analog Chip Moat", "BRK-B": "Fortress Balance Sheet", "RKLB": "Space Infrastructure"} 

DEFAULT_PORTFOLIO = {"BRK-B": 1361.56, "NVDA": 1128.72, "JNJ": 745.42, "TSLA": 475.33, "ENPH": 311.58, "TXN": 308.51, "RKLB": 156.75}

st.set_page_config(page_title="QuantHQ DCA", page_icon="🛡️", layout="wide")
st.title("🛡️ QUANT-HQ DCA (Institutional Execution)")

# --- Session State Management ---
if 'entry_candidates_history' not in st.session_state: st.session_state['entry_candidates_history'] = {}
if 'rotation_history' not in st.session_state: st.session_state['rotation_history'] = {}

# ==========================================
# 🎛️ UI & SIDEBAR
# ==========================================
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

# ==========================================
# 🚀 CORE ENGINE
# ==========================================
if st.button("🚀 รันระบบ Quant Matrix", type="primary"):
    st.session_state['run_quant_engine'] = True
    st.session_state['matrix_calculated'] = False 

if st.session_state.get('run_quant_engine', False):
    if not st.session_state.get('matrix_calculated', False):
        status_box = st.status("🔮 กำลังเดินเครื่องสถาปัตยกรรม...", expanded=True)
        
        market_proxy = [t for sublist in SECTOR_DB.values() for t in sublist]
        total_universe = list(set(my_portfolio + market_proxy))
        
        pipeline = InstitutionalDataPipeline(total_universe)
        regime_engine = MarketRegimeHMM(n_states=2)
        factor_allocator = DynamicFactorAllocator()
        
        # --- Layer 3: Regime ---
        status_box.update(label="📡 Layer 3: ดึงข้อมูล Macro และคำนวณ HMM...")
        df_macro = regime_engine.fetch_macro_features()
        raw_probs = regime_engine.expanding_fit_predict(df_macro)
        smooth_probs = regime_engine.apply_transition_smoothing(raw_probs)
        hysteresis_probs = regime_engine.apply_hysteresis(smooth_probs)
        regime_weights = factor_allocator.calculate_weights(hysteresis_probs)
        P_PANIC = regime_weights.get('P_PANIC', 0.0)
        
        # --- Layer 1: Data ---
        status_box.update(label="🧼 Layer 1: ร่อนหุ้นผ่าน Sanity Check Pipeline...")
        clean_universe = pipeline.filter_universe()
        final_scan_list = list(set(my_portfolio + clean_universe))
        
        raw_prices = pipeline.fetch_bulk_market_data(final_scan_list)
        prices_1y = raw_prices.tail(252)
        returns_1y = prices_1y.pct_change().fillna(0)
        max_dd = ((prices_1y - prices_1y.cummax()) / prices_1y.cummax()).min() * 100
        
        # --- Layer 2: Factors ---
        status_box.update(label="🧬 Layer 2: สกัด Factor และคำนวณ Z-Score...")
        from data_loader import fetch_fundamental_data
        df_fundamentals = fetch_fundamental_data(final_scan_list)
        df_clean_fundamentals = pipeline.clean_fundamentals(df_fundamentals)
        
        metrics = []
        rsi_data, sr_data = {}, {}
        spy_ret = df_macro['SPY_Ret'].tail(252).reindex(returns_1y.index).fillna(0)
        
        # 🟢 อัปเดต Bug #1: Dynamic Alpha Decay
        if os.path.exists(PORTFOLIO_FILE):
            last_mod_time = datetime.fromtimestamp(os.path.getmtime(PORTFOLIO_FILE))
            days_since_last = max(1, (datetime.now() - last_mod_time).days)
        else:
            days_since_last = 3
        
        for t in final_scan_list:
            try:
                cov = returns_1y[t].cov(spy_ret)
                var = spy_ret.var()
                beta = cov / var if var > 0 else 1.0
                
                s_t = prices_1y[t].dropna()
                
                # 🔴 อัปเดต Bug #2: Momentum Skip t-1 (หลบ Short-term Reversal)
                if len(s_t) > 147: 
                    ret_6m = (s_t.iloc[-22] / s_t.iloc[-147]) - 1
                else:
                    ret_6m = 0.0
                    
                mkt_ret_6m = (1 + spy_ret.iloc[-147:-22]).prod() - 1 if len(spy_ret) > 147 else 0.0
                residual_mom = ret_6m - (beta * mkt_ret_6m)
            except: 
                residual_mom, beta = 0.0, 1.0
            
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
        
        # ==========================================
        # 🟡 SHOWCASE #6: Per-factor Alpha Decay 
        # ==========================================
        # สลายตัว Z-Score แต่ละตัวตาม Half-life ที่ต่างกัน ก่อนนำไปรวม
        decayed_mom = z_mom.apply(lambda x: calculate_alpha_decay(x, days_passed=days_since_last, half_life_days=21))
        decayed_val = z_value.apply(lambda x: calculate_alpha_decay(x, days_passed=days_since_last, half_life_days=45))
        decayed_qual = z_quality.apply(lambda x: calculate_alpha_decay(x, days_passed=days_since_last, half_life_days=90))
        
        w_m, w_q, w_v = regime_weights['Mom'], regime_weights['Qual'], regime_weights['Val']
        
        # ผสม Factor ที่ผ่านการทำ Decay แล้วเข้าด้วยกัน
        df_metrics['Alpha_Score'] = (decayed_mom * w_m) + (decayed_qual * w_q) + (decayed_val * w_v)
        df_metrics['Max_Drawdown'] = df_metrics['Ticker'].map(max_dd)
        
        final_df = df_metrics.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
        current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(len(port_df))

        # --- Layer 4: Optimizer ---
        status_box.update(label="🏛️ Layer 4: Optimizer & Micro-Execution Gate...")
        if "Auto-Pilot" in engine_choice:
            opt_res, _ = run_institutional_black_litterman(
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

        port_df['Beta'] = port_df['Ticker'].map(dict(zip(df_metrics['Ticker'], df_metrics['Beta'])))
        port_df['Sell_Amount'] = 0.0
        port_df['Action_Reason'] = "⚖️ ถือรักษาสมดุล (รอรอบใหม่)" 

        # ==========================================
        # 🛑 DECISION 3: EXIT RULES (สั่ง Action จริง)
        # ==========================================
        def evaluate_exit_signals(df, p_panic):
            exit_signals = []
            for _, row in df.iterrows():
                t = row['Ticker']
                reasons = []
                severity = 'HOLD'
                
                # Alpha แย่ลด 30%
                if row['Alpha_Score'] < -0.5:
                    reasons.append(f"Alpha ติดลบ ({row['Alpha_Score']:.2f})")
                    severity = 'REDUCE'
                
                # Regime Panic + หุ้นซิ่ง + Alpha แย่ = หนี
                beta = row.get('Beta', 1.0)
                if p_panic > 0.7 and beta > 1.5 and row['Alpha_Score'] < 0:
                    reasons.append(f"PANIC + หุ้นผันผวนสูง + Alpha แย่")
                    severity = 'EXIT'
                    
                if reasons:
                    exit_signals.append({'Ticker': t, 'Severity': severity, 'Reasons': reasons})
            return exit_signals

        st.session_state['exit_signals'] = evaluate_exit_signals(port_df, P_PANIC)
        
        for sig in st.session_state['exit_signals']:
            idx = port_df['Ticker'] == sig['Ticker']
            if sig['Severity'] == 'EXIT':
                port_df.loc[idx, 'Target_%'] = 0.0
                port_df.loc[idx, 'Sell_Amount'] = port_df.loc[idx, 'Current']
                port_df.loc[idx, 'Action_Reason'] = f"🔴 ขายทิ้ง ({sig['Reasons'][0]})"
            elif sig['Severity'] == 'REDUCE':
                port_df.loc[idx, 'Target_%'] *= 0.70  
                port_df.loc[idx, 'Sell_Amount'] = port_df.loc[idx, 'Current'] * 0.30
                port_df.loc[idx, 'Action_Reason'] = f"🟡 ลดสัดส่วน 30% ({sig['Reasons'][0]})"

        # ==========================================
        # 🛡️ Threshold Rebalancing 
        # ==========================================
        MIN_DEVIATION = 5.0  
        mask_small_diff = (port_df['Target_%'] - port_df['Weight_%']).abs() < MIN_DEVIATION
        mask_not_selling = port_df['Sell_Amount'] == 0
        
        port_df.loc[mask_small_diff & mask_not_selling, 'Target_%'] = port_df.loc[mask_small_diff & mask_not_selling, 'Weight_%']
        port_df.loc[mask_small_diff & mask_not_selling, 'Action_Reason'] = "🔒 น้ำหนักไม่ถึงเกณฑ์สับเปลี่ยน (5%)"
        
        total_target = port_df['Target_%'].sum()
        if total_target > 0: port_df['Target_%'] = (port_df['Target_%'] / total_target) * 100.0

        port_df['Target_Val'] = (total_port_value + actual_budget) * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current']
        port_df['Buy_Amount'] = 0.0

        # ==========================================
        # 🟢 DECISION 1: DCA Allocation & เหตุผล
        # ==========================================
        port_df['Regime_Weight'] = 1.0 - (P_PANIC * port_df['Beta'].clip(0, 2) / 2)
        port_df['Weighted_Deficit'] = port_df['Deficit'] * port_df['Regime_Weight']
        
        buy_mask = (port_df['Weighted_Deficit'] > min_order_thb) & (port_df['Alpha_Score'] > 0) & mask_not_selling
        mask_no_buy = (~buy_mask) & mask_not_selling
        
        port_df.loc[mask_no_buy & (port_df['Alpha_Score'] <= 0), 'Action_Reason'] = "🚫 งดซื้อ (Alpha ติดลบ)"
        port_df.loc[mask_no_buy & (port_df['Alpha_Score'] > 0) & (port_df['Weighted_Deficit'] <= min_order_thb), 'Action_Reason'] = "🎯 ทุนเต็มเป้าหมายแล้ว"
        
        sum_def = port_df.loc[buy_mask, 'Weighted_Deficit'].sum()
        if sum_def > 0: 
            port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Weighted_Deficit'] / sum_def) * actual_budget
            port_df.loc[buy_mask, 'Action_Reason'] = "🟢 ซื้อสะสม (DCA Approved)"
        
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ', 'Sell_Amount': 'ขาย', 'Action_Reason': 'เหตุผล (Action)'})
        out['MDD'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk)
        out['รับ/ต้าน'] = out['หุ้น'].map(sr_data)
        
        top_alpha_display = final_df.head(10).copy() 
        top_alpha_display['Risk_Adj_Alpha'] = (top_alpha_display['Alpha_Score'] / (top_alpha_display['Max_Drawdown'].abs() + 1e-9)).round(3)
        top_alpha_display['Alpha_Score'] = top_alpha_display['Alpha_Score'].round(2)
        top_alpha_display['MDD'] = top_alpha_display['Max_Drawdown'].round(1).astype(str) + "%"
        top_alpha_display['สถานะ'] = top_alpha_display['Ticker'].apply(lambda x: "💼 ถืออยู่" if x in my_portfolio else "✨ เป้าหมายใหม่")

        # ==========================================
        # 🌟 DECISION 2: New Position 
        # ==========================================
        def evaluate_new_entries(candidates_df, p_df, p_panic, m_port):
            if len(m_port) >= 10: return []
            avg_a = p_df['Alpha_Score'].mean() if not p_df.empty else 0
            worst_r = (p_df['Alpha_Score'] / (p_df['Max_Drawdown'].abs() + 1e-9)).min() if not p_df.empty else -999
            
            cands = candidates_df[candidates_df['สถานะ'] == "✨ เป้าหมายใหม่"]
            passed = []
            for _, r in cands.iterrows():
                if (r['Alpha_Score'] > avg_a + 1.0) and (r['Risk_Adj_Alpha'] > worst_r) and not (p_panic > 0.5 and float(r['MDD'].replace('%','')) < -30):
                    passed.append(r['Ticker'])
            return passed

        new_entries = evaluate_new_entries(top_alpha_display, port_df, P_PANIC, my_portfolio)
        for t in new_entries: st.session_state['entry_candidates_history'][t] = st.session_state['entry_candidates_history'].get(t, 0) + 1
        for t in list(st.session_state['entry_candidates_history'].keys()):
            if t not in new_entries: st.session_state['entry_candidates_history'][t] = 0
        st.session_state['confirmed_new_entries'] = [t for t, c in st.session_state['entry_candidates_history'].items() if c >= 2]

        # ==========================================
        # 🔄 DECISION 4: Rotation 
        # ==========================================
        def evaluate_rotation(top_a_disp, p_panic, budget):
            cur = top_a_disp[top_a_disp['สถานะ'] == "💼 ถืออยู่"]
            out = top_a_disp[top_a_disp['สถานะ'] == "✨ เป้าหมายใหม่"]
            if cur.empty or out.empty: return None
            w_held = cur.sort_values('Risk_Adj_Alpha').iloc[0]
            b_new = out.sort_values('Risk_Adj_Alpha', ascending=False).iloc[0]
            
            a_diff = b_new['Alpha_Score'] - w_held['Alpha_Score']
            r_diff = b_new['Risk_Adj_Alpha'] - w_held['Risk_Adj_Alpha']
            if a_diff < 1.0 or r_diff < 0.005 or p_panic > 0.6: return None
            
            est_pos = budget * 10
            if (a_diff * 0.02 * est_pos) < (est_pos * 0.001 * 2 * 3): return None
            return {'buy': b_new['Ticker'], 'sell': w_held['Ticker'], 'alpha_diff': a_diff}

        rot_signal = evaluate_rotation(top_alpha_display, P_PANIC, actual_budget)
        if rot_signal:
            k = f"{rot_signal['buy']}_vs_{rot_signal['sell']}"
            st.session_state['rotation_history'][k] = st.session_state['rotation_history'].get(k, 0) + 1
            rot_signal['status'] = 'CONFIRMED' if st.session_state['rotation_history'][k] >= 2 else 'PENDING'
            st.session_state['rotation_alert'] = rot_signal
        else: st.session_state['rotation_alert'] = None

        # Data for Display
        t_exposure = port_df.groupby('Sector')['Target_%'].sum().reset_index()
        p_state = json.dumps({"market_regime": f"{regime_weights['Current_State']}", "proposed_buys": out[out['ซื้อ']>0][['หุ้น', 'ซื้อ']].to_dict('records')})

        st.session_state['regime_report'] = f"📊 HMM Regime -> State: {regime_weights['Current_State']} | 🐂 P(Bull): {regime_weights['P_BULL']*100:.1f}% | 🐻 P(Panic): {regime_weights['P_PANIC']*100:.1f}%"
        st.session_state['factor_mix'] = f"⚡ Dynamic Factors -> Mom: {w_m*100:.0f}% | Qual: {w_q*100:.0f}% | Val: {w_v*100:.0f}%"
        st.session_state['out_table'] = out
        st.session_state['top_alpha_table'] = top_alpha_display
        st.session_state['sector_exposure'] = t_exposure
        st.session_state['p_state_json'] = p_state
        
        status_box.update(label="--- 4 Layers Framework ประมวลผลเสร็จสิ้น ---", state="complete")
        st.session_state['matrix_calculated'] = True
        
    st.info(st.session_state['regime_report'])
    st.success(st.session_state['factor_mix'])
    
    # ==========================================
    # 📢 ALERTS (UI)
    # ==========================================
    if st.session_state.get('exit_signals'):
        for sig in st.session_state['exit_signals']:
            if sig['Severity'] == 'EXIT': st.error(f"**🔴 EXIT SIGNAL:** {sig['Ticker']} — {' | '.join(sig['Reasons'])}")
            elif sig['Severity'] == 'REDUCE': st.warning(f"**🟡 REDUCE SIGNAL:** {sig['Ticker']} — {' | '.join(sig['Reasons'])}")

    if st.session_state.get('confirmed_new_entries'):
        for t in st.session_state['confirmed_new_entries']:
            st.success(f"**🟢 NEW POSITION CONFIRMED:** หุ้น {t} ผ่านเกณฑ์ 3 ข้อต่อเนื่อง พิจารณาเพิ่มเข้าพอร์ต!")

    if st.session_state.get('rotation_alert'):
        rot = st.session_state['rotation_alert']
        if rot['status'] == 'CONFIRMED': st.success(f"**🔄 CONFIRMED ROTATION:** ขาย **{rot['sell']}** เข้า **{rot['buy']}** (Alpha Diff: +{rot['alpha_diff']:.2f})")
        else: st.info(f"**⏳ PENDING ROTATION:** เล็งสับเปลี่ยน **{rot['sell']}** เข้า **{rot['buy']}** (รอรอบยืนยันถัดไป)")

    # ==========================================
    # 📋 MAIN TABLES & CHARTS (UI)
    # ==========================================
    st.markdown("### 📋 2. ตาราง Quant Allocation (Execution Log)")
    st.dataframe(st.session_state['out_table'][['หุ้น', 'เหตุผล (Action)', 'MDD', 'RSI', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']].round(2).sort_values('ซื้อ', ascending=False), use_container_width=True, hide_index=True)
    
    # 📊 PRIORITY 3: SECTOR EXPOSURE CHART
    st.markdown("### 📊 3. สัดส่วนอุตสาหกรรมเป้าหมาย (Target Sector Exposure)")
    fig = px.bar(st.session_state['sector_exposure'], x='Target_%', y='Sector', orientation='h', text='Target_%', color='Target_%', color_continuous_scale='Teal')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Target Weight (%)", yaxis_title="", showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 📡 [RADAR] TOP ALPHA CANDIDATES")
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
