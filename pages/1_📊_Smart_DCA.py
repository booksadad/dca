import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json
import warnings
from datetime import datetime
import plotly.express as px
import shutil

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
LOG_FILE = "trade_log.csv"

# ==========================================
# 💾 ฐานข้อมูลและ Baseline
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
    st.session_state['logged_this_run'] = False 

if st.session_state.get('run_quant_engine', False):
    if not st.session_state.get('matrix_calculated', False):
        status_box = st.status("🔮 กำลังเดินเครื่องสถาปัตยกรรม...", expanded=True)
        
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
        
        # ==========================================
        # 🛡️ GUARDRAIL 1: Data Freshness Check
        # ==========================================
        last_date = raw_prices.index[-1]
        today = pd.Timestamp.now(tz='US/Eastern').normalize()
        bdays_behind = np.busday_count(last_date.date(), today.date())
        if bdays_behind > 2:
            st.error(f"🚨 สัญญาณอันตราย! ข้อมูลตลาดเก่าเกินไป ({bdays_behind} วันทำการ) ระบบระงับการออกคำสั่งเทรดเพื่อความปลอดภัย!")
            st.stop()

        # ==========================================
        # 🛡️ GUARDRAIL 2: FX Rate Fetching
        # ==========================================
        status_box.update(label="💱 ดึงอัตราแลกเปลี่ยน (USD/THB) ล่าสุด...")
        try:
            fx_data = yf.download("THB=X", period="5d", progress=False)
            current_fx = float(fx_data['Close'].dropna().iloc[-1])
        except:
            current_fx = 36.50 # Fallback ยามฉุกเฉิน
            st.warning(f"⚠️ ดึงค่าเงินไม่สำเร็จ ใช้ค่าประมาณ {current_fx} THB/USD")
        st.session_state['current_fx'] = current_fx

        prices_1y = raw_prices.tail(252)
        returns_1y = prices_1y.pct_change().fillna(0)
        max_dd = ((prices_1y - prices_1y.cummax()) / prices_1y.cummax()).min() * 100
        
        current_prices_dict = prices_1y.iloc[-1].to_dict()
        st.session_state['current_prices'] = current_prices_dict
        
        status_box.update(label="🧬 Layer 2: สกัด Factor และคำนวณ Z-Score...")
        from data_loader import fetch_fundamental_data
        df_fundamentals = fetch_fundamental_data(final_scan_list)
        df_clean_fundamentals = pipeline.clean_fundamentals(df_fundamentals)
        
        metrics = []
        rsi_data, sr_data = {} , {}
        spy_ret = df_macro['SPY_Ret'].tail(252).reindex(returns_1y.index).fillna(0)
        
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
        
        decayed_mom = z_mom.apply(lambda x: calculate_alpha_decay(x, days_passed=days_since_last, half_life_days=21))
        decayed_val = z_value.apply(lambda x: calculate_alpha_decay(x, days_passed=days_since_last, half_life_days=45))
        decayed_qual = z_quality.apply(lambda x: calculate_alpha_decay(x, days_passed=days_since_last, half_life_days=90))
        
        w_m, w_q, w_v = regime_weights['Mom'], regime_weights['Qual'], regime_weights['Val']
        
        df_metrics['Alpha_Score'] = (decayed_mom * w_m) + (decayed_qual * w_q) + (decayed_val * w_v)
        df_metrics['Max_Drawdown'] = df_metrics['Ticker'].map(max_dd)
        
        final_df = df_metrics.sort_values(by='Alpha_Score', ascending=False).reset_index(drop=True)
        port_df = final_df[final_df['Ticker'].isin(my_portfolio)].copy()
        port_df['Current'] = port_df['Ticker'].map(current_thb)
        
        total_port_value = port_df['Current'].sum()
        port_df['Weight_%'] = (port_df['Current'] / total_port_value) * 100 if total_port_value > 0 else 0
        current_weights_arr = (port_df['Current'] / total_port_value).fillna(0).values if total_port_value > 0 else np.zeros(len(port_df))

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

        def evaluate_exit_signals(df, p_panic):
            exit_signals = []
            for _, row in df.iterrows():
                t = row['Ticker']
                reasons = []
                severity = 'HOLD'
                if row['Alpha_Score'] < -0.5:
                    reasons.append(f"Alpha ติดลบ ({row['Alpha_Score']:.2f})")
                    severity = 'REDUCE'
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

        MIN_DEVIATION = 5.0  
        mask_small_diff = (port_df['Target_%'] - port_df['Weight_%']).abs() < MIN_DEVIATION
        mask_not_selling = port_df['Sell_Amount'] == 0
        port_df.loc[mask_small_diff & mask_not_selling, 'Target_%'] = port_df.loc[mask_small_diff & mask_not_selling, 'Weight_%']
        port_df.loc[mask_small_diff & mask_not_selling, 'Action_Reason'] = "🔒 น้ำหนักไม่ถึงเกณฑ์สับเปลี่ยน (5%)"
        
        # ==========================================
        # 🛡️ GUARDRAIL 3: Concentration Limit (Hard Stop ที่ 35%)
        # ==========================================
        MAX_WEIGHT_LIMIT = 35.0
        port_df['Target_%'] = port_df['Target_%'].clip(upper=MAX_WEIGHT_LIMIT)
        
        total_target = port_df['Target_%'].sum()
        if total_target > 0: port_df['Target_%'] = (port_df['Target_%'] / total_target) * 100.0

        port_df['Target_Val'] = (total_port_value + actual_budget) * (port_df['Target_%'] / 100)
        port_df['Deficit'] = port_df['Target_Val'] - port_df['Current']
        port_df['Buy_Amount'] = 0.0

        # ==========================================
        # 🟢 DECISION 1 & Commission Check
        # ==========================================
        port_df['Regime_Weight'] = 1.0 - (P_PANIC * port_df['Beta'].clip(0, 2) / 2)
        port_df['Weighted_Deficit'] = port_df['Deficit'] * port_df['Regime_Weight']
        
        buy_mask = (port_df['Weighted_Deficit'] > 0) & (port_df['Alpha_Score'] > 0) & mask_not_selling
        mask_no_buy = (~buy_mask) & mask_not_selling
        
        port_df.loc[mask_no_buy & (port_df['Alpha_Score'] <= 0), 'Action_Reason'] = "🚫 งดซื้อ (Alpha ติดลบ)"
        port_df.loc[mask_no_buy & (port_df['Alpha_Score'] > 0), 'Action_Reason'] = "🎯 ทุนเต็มเป้าหมายแล้ว"
        
        sum_def = port_df.loc[buy_mask, 'Weighted_Deficit'].sum()
        if sum_def > 0: 
            port_df.loc[buy_mask, 'Buy_Amount'] = (port_df.loc[buy_mask, 'Weighted_Deficit'] / sum_def) * actual_budget
            
            # --- กรอง Commission < 2% ---
            broker_fee_usd = 0.15 # สมมติใช้ Dime! ($0.15 ต่อไม้) เปลี่ยนค่าได้ถ้าย้ายโบรค
            commission_thb = broker_fee_usd * current_fx
            
            for idx in port_df[buy_mask].index:
                amt = port_df.loc[idx, 'Buy_Amount']
                if amt < min_order_thb:
                    port_df.loc[idx, 'Buy_Amount'] = 0.0
                    port_df.loc[idx, 'Action_Reason'] = f"🚫 ยอดซื้อไม่ถึงขั้นต่ำ ({min_order_thb}฿)"
                elif (commission_thb / amt) > 0.02:
                    port_df.loc[idx, 'Buy_Amount'] = 0.0
                    port_df.loc[idx, 'Action_Reason'] = f"🚫 สั่งระงับ (โดนค่าคอมกินเกิน 2%)"
                else:
                    port_df.loc[idx, 'Action_Reason'] = "🟢 ซื้อสะสม (ผ่านเกณฑ์ค่าคอม)"
        
        out = port_df.copy().rename(columns={'Ticker': 'หุ้น', 'Target_%': 'เป้า%', 'Current': 'ทุนเดิม', 'Buy_Amount': 'ซื้อ', 'Sell_Amount': 'ขาย', 'Action_Reason': 'เหตุผล (Action)'})
        out['MDD'] = out['Max_Drawdown'].apply(lambda x: f"{x:.1f}%")
        out['RSI'] = out['หุ้น'].map(rsi_data).apply(check_doi_risk)
        out['รับ/ต้าน'] = out['หุ้น'].map(sr_data)
        
        top_alpha_display = final_df.head(10).copy() 
        top_alpha_display['Risk_Adj_Alpha'] = (top_alpha_display['Alpha_Score'] / (top_alpha_display['Max_Drawdown'].abs() + 1e-9)).round(3)
        top_alpha_display['Alpha_Score'] = top_alpha_display['Alpha_Score'].round(2)
        top_alpha_display['MDD'] = top_alpha_display['Max_Drawdown'].round(1).astype(str) + "%"
        top_alpha_display['สถานะ'] = top_alpha_display['Ticker'].apply(lambda x: "💼 ถืออยู่" if x in my_portfolio else "✨ เป้าหมายใหม่")

        t_exposure = port_df.groupby('Sector')['Target_%'].sum().reset_index()
        p_state = json.dumps({"market_regime": f"{regime_weights['Current_State']}", "proposed_buys": out[out['ซื้อ']>0][['หุ้น', 'ซื้อ']].to_dict('records')})
        
        st.session_state['current_regime_weights'] = regime_weights
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
    
    if st.session_state.get('exit_signals'):
        for sig in st.session_state['exit_signals']:
            if sig['Severity'] == 'EXIT': st.error(f"**🔴 EXIT SIGNAL:** {sig['Ticker']} — {' | '.join(sig['Reasons'])}")
            elif sig['Severity'] == 'REDUCE': st.warning(f"**🟡 REDUCE SIGNAL:** {sig['Ticker']} — {' | '.join(sig['Reasons'])}")

    # ==========================================
    # 📈 PHASE 1: PERFORMANCE MEASUREMENT (P&L Dashboard)
    # ==========================================
    def calculate_portfolio_performance(log_file, current_prices):
        if not os.path.exists(log_file): return None
        df_log = pd.read_csv(log_file)
        if 'Shares' not in df_log.columns: return None # กรณีไฟล์เก่าที่ยังไม่มีหุ้น

        perf_data = []
        for ticker in df_log['Ticker'].unique():
            ticker_trades = df_log[df_log['Ticker'] == ticker].dropna(subset=['Shares'])
            total_shares = 0.0
            total_cost = 0.0
            for _, trade in ticker_trades.iterrows():
                if trade['Action'] == 'BUY':
                    total_shares += trade['Shares']
                    total_cost += trade['Amount_THB']
                elif trade['Action'] == 'SELL':
                    if total_shares > 0:
                        avg_cost_temp = total_cost / total_shares
                        total_shares -= trade['Shares']
                        total_cost -= (avg_cost_temp * trade['Shares'])
                        
            if total_shares > 1e-6:
                avg_cost = total_cost / total_shares
                cur_price = current_prices.get(ticker, avg_cost)
                market_value = total_shares * cur_price
                unrealized_pl = market_value - total_cost
                unrealized_pl_pct = (unrealized_pl / total_cost) * 100 if total_cost > 0 else 0
                
                perf_data.append({
                    'Ticker': ticker,
                    'Shares': round(total_shares, 4),
                    'Avg_Cost': round(avg_cost, 2),
                    'Total_Cost': round(total_cost, 2),
                    'Market_Value': round(market_value, 2),
                    'Unrealized_P&L': round(unrealized_pl, 2),
                    'P&L_%': round(unrealized_pl_pct, 2)
                })
        return pd.DataFrame(perf_data)

    current_prices = st.session_state.get('current_prices', {})
    perf_df = calculate_portfolio_performance(LOG_FILE, current_prices)
    
    if perf_df is not None and not perf_df.empty:
        st.markdown("### 💰 กระจกสะท้อนผลงานพอร์ต (Portfolio P&L)")
        total_investment = perf_df['Total_Cost'].sum()
        total_market_value = perf_df['Market_Value'].sum()
        total_pl = total_market_value - total_investment
        total_pl_pct = (total_pl / total_investment) * 100 if total_investment > 0 else 0
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("มูลค่าพอร์ตปัจจุบัน (Market Value)", f"฿ {total_market_value:,.2f}")
        mc2.metric("ต้นทุนสะสม (Total Cost)", f"฿ {total_investment:,.2f}")
        mc3.metric("กำไร/ขาดทุนสะสม (Unrealized P&L)", f"฿ {total_pl:,.2f}", f"{total_pl_pct:,.2f}%")
        
        # จัด Format ตารางให้ดูง่าย มีสีสัน
        # จัด Format ตารางให้ดูง่าย มีสีสัน
        styled_perf = perf_df.style.map(lambda x: 'color: #2ca02c' if x > 0 else 'color: #d62728' if x < 0 else '', subset=['Unrealized_P&L', 'P&L_%']).format({'Avg_Cost': '฿{:.2f}', 'Total_Cost': '฿{:.2f}', 'Market_Value': '฿{:.2f}', 'Unrealized_P&L': '฿{:.2f}', 'P&L_%': '{:.2f}%'})
        st.dataframe(styled_perf, use_container_width=True, hide_index=True)
        st.markdown("---")

    # ==========================================
    # 📋 MAIN TABLES & CHARTS
    # ==========================================
    st.markdown("### 📋 ตาราง Quant Allocation (คำสั่ง Execution รอบนี้)")
    st.dataframe(st.session_state['out_table'][['หุ้น', 'เหตุผล (Action)', 'MDD', 'RSI', 'เป้า%', 'ทุนเดิม', 'ซื้อ', 'ขาย']].round(2).sort_values('ซื้อ', ascending=False), use_container_width=True, hide_index=True)
    
    st.markdown("### 📊 สัดส่วนอุตสาหกรรมเป้าหมาย (Target Sector Exposure)")
    fig = px.bar(st.session_state['sector_exposure'], x='Target_%', y='Sector', orientation='h', text='Target_%', color='Target_%', color_continuous_scale='Teal')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Target Weight (%)", yaxis_title="", showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 📡 [RADAR] TOP ALPHA CANDIDATES")
    st.dataframe(st.session_state['top_alpha_table'][['Ticker', 'Sector', 'Alpha_Score', 'Risk_Adj_Alpha', 'MDD', 'สถานะ']].rename(columns={'Ticker': 'หุ้น', 'Alpha_Score': 'Alpha Score', 'Risk_Adj_Alpha': 'Risk-adj Alpha', 'MDD': 'Max Drawdown'}), use_container_width=True, hide_index=True)

    # ==========================================
    # 📝 TRADE LOGGING SYSTEM (อัปเกรดเก็บ Price & Shares)
    # ==========================================
    st.markdown("---")
    st.subheader("💾 บันทึกประวัติการทำรายการ (Trade Logger)")
    
    if st.button("✅ ยืนยันคำสั่งและบันทึกประวัติลง CSV", type="primary"):
        if not st.session_state.get('logged_this_run', False):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            action_df = st.session_state['out_table']
            trades = action_df[(action_df['ซื้อ'] > 0) | (action_df['ขาย'] > 0)].copy()
            regime_data = st.session_state.get('current_regime_weights', {})
            prices_dict = st.session_state.get('current_prices', {})
            
            if not trades.empty:
                log_data = []
                for _, row in trades.iterrows():
                    alpha_lookup = round(st.session_state['top_alpha_table'].set_index('Ticker')['Alpha_Score'].get(row['หุ้น'], 0), 3)
                    exec_price = prices_dict.get(row['หุ้น'], 1.0)
                    
                    if row['ซื้อ'] > 0:
                        shares_bought = round(row['ซื้อ'] / exec_price, 6)
                        log_data.append({'Date': current_time, 'Ticker': row['หุ้น'], 'Action': 'BUY', 'Amount_THB': round(row['ซื้อ'], 2), 'Price': round(exec_price, 2), 'Shares': shares_bought, 'Reason': row['เหตุผล (Action)'], 'Regime': regime_data.get('Current_State', 'N/A'), 'P_Bull': round(regime_data.get('P_BULL', 0) * 100, 1), 'P_Panic': round(regime_data.get('P_PANIC', 0) * 100, 1), 'Alpha_Score': alpha_lookup})
                    if row['ขาย'] > 0:
                        shares_sold = round(row['ขาย'] / exec_price, 6)
                        log_data.append({'Date': current_time, 'Ticker': row['หุ้น'], 'Action': 'SELL', 'Amount_THB': round(row['ขาย'], 2), 'Price': round(exec_price, 2), 'Shares': shares_sold, 'Reason': row['เหตุผล (Action)'], 'Regime': regime_data.get('Current_State', 'N/A'), 'P_Bull': round(regime_data.get('P_BULL', 0) * 100, 1), 'P_Panic': round(regime_data.get('P_PANIC', 0) * 100, 1), 'Alpha_Score': alpha_lookup})
                
                new_log_df = pd.DataFrame(log_data)
                
                tmp_log = LOG_FILE + ".tmp"
                if os.path.exists(LOG_FILE):
                    existing_log = pd.read_csv(LOG_FILE)
                    updated_log = pd.concat([existing_log, new_log_df], ignore_index=True)
                else:
                    updated_log = new_log_df
                updated_log.to_csv(tmp_log, index=False)
                shutil.move(tmp_log, LOG_FILE)
                
                if os.path.exists(PORTFOLIO_FILE):
                    port_df_current = pd.read_csv(PORTFOLIO_FILE)
                    for _, row in trades.iterrows():
                        t = row['หุ้น']
                        net_change = row['ซื้อ'] - row['ขาย']
                        
                        idx = port_df_current['รายชื่อหุ้น'] == t
                        if idx.any():
                            port_df_current.loc[idx, 'ยอดเงินปัจจุบัน (บาท)'] += net_change
                        elif row['ซื้อ'] > 0:
                            new_row = pd.DataFrame([{'รายชื่อหุ้น': t, 'ยอดเงินปัจจุบัน (บาท)': row['ซื้อ']}])
                            port_df_current = pd.concat([port_df_current, new_row], ignore_index=True)
                            
                    tmp_port = PORTFOLIO_FILE + ".tmp"
                    port_df_current.to_csv(tmp_port, index=False)
                    shutil.move(tmp_port, PORTFOLIO_FILE)

                st.session_state['logged_this_run'] = True
                st.success(f"💾 บันทึกประวัติพร้อมคำนวณ Price/Shares สำเร็จ {len(new_log_df)} รายการ!")
                st.rerun() 
            else:
                st.session_state['logged_this_run'] = True
                st.info("รอบนี้ไม่มีคำสั่งซื้อขายที่ต้องบันทึกครับ (Hold รักษาสมดุล)")
        else:
            st.warning("⚠️ รอบนี้บันทึกไปแล้วครับ — กด 'รันระบบ' ใหม่ก่อนบันทึกรอบถัดไป")

    if os.path.exists(LOG_FILE):
        with st.expander("📂 ดูประวัติการเทรดย้อนหลังทั้งหมด (Trade Log)"):
            history_df = pd.read_csv(LOG_FILE)
            col1, col2, col3 = st.columns(3)
            col1.metric("รายการทั้งหมด", len(history_df))
            col2.metric("ซื้อสะสม (บาท)", f"{history_df[history_df['Action']=='BUY']['Amount_THB'].sum():,.0f}")
            col3.metric("ขายสะสม (บาท)", f"{history_df[history_df['Action']=='SELL']['Amount_THB'].sum():,.0f}")
            st.dataframe(history_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
            
            # --- ปุ่มสำหรับกด Reset ล้างไพ่ ---
            st.markdown("---")
            if st.button("🗑️ ล้างประวัติการเทรดทั้งหมด (Reset Trade Log)", type="secondary"):
                os.remove(LOG_FILE)
                st.session_state['logged_this_run'] = False
                st.success("ล้างประวัติการเทรดเรียบร้อยแล้ว!")
                st.rerun()
