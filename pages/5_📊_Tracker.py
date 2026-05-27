import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
import os

warnings.filterwarnings("ignore")
PORTFOLIO_FILE = "my_portfolio_data.csv"

st.set_page_config(page_title="Walk-Forward Backtest", page_icon="🔬", layout="wide")
st.title("🔬 QUANT-HQ: Walk-Forward Backtester")
st.markdown("ระบบทดสอบย้อนหลังระดับสถาบัน **(No Lookahead Bias + Transaction Costs)**")
st.markdown("---")

# ==========================================
# 1. โหลดรายชื่อหุ้นจากพอร์ตปัจจุบัน
# ==========================================
my_portfolio = []
if os.path.exists(PORTFOLIO_FILE):
    try:
        temp_df = pd.read_csv(PORTFOLIO_FILE)
        if not temp_df.empty and "รายชื่อหุ้น" in temp_df.columns:
            my_portfolio = [str(t).strip().upper() for t in temp_df["รายชื่อหุ้น"].tolist() if str(t).strip()]
    except: pass

if not my_portfolio:
    st.warning("⚠️ ไม่พบหุ้นในพอร์ต โปรดไปตั้งค่าพอร์ตที่หน้า Smart DCA ก่อนครับ")
    st.stop()

# ==========================================
# 2. ตั้งค่า Parameters สำหรับ Backtest
# ==========================================
st.sidebar.markdown(f"**💼 หุ้นใน Universe:**\n{', '.join(my_portfolio)}")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Backtest Parameters")
lookback_years = st.sidebar.slider("ระยะเวลาทดสอบ (ปี)", 1, 5, 3)
train_window = 252 # ใช้ข้อมูล 1 ปีฝึกโมเดล
step_size = 21 # Rebalance ทุกๆ 1 เดือน (21 วันทำการ)

st.sidebar.subheader("💸 Transaction Costs")
broker_fee = st.sidebar.number_input("ค่าคอมมิชชัน (%)", 0.0, 1.0, 0.15) / 100
slippage = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.10) / 100
total_t_cost = broker_fee + slippage

if st.button("🚀 รันระบบ Walk-Forward Simulation", type="primary"):
    with st.status("⏳ กำลังจำลองการเดินทางข้ามเวลา (Walk-Forward)...", expanded=True) as status:
        
        benchmark = "VOO"
        tickers_to_fetch = list(set(my_portfolio + [benchmark]))
        
        status.update(label="📡 กำลังดึงข้อมูล Time-Series ย้อนหลัง...")
        data = yf.download(tickers_to_fetch, period=f"{lookback_years + 1}y", progress=False)['Close'].dropna()
        
        if len(data) < train_window + step_size:
            st.error("❌ ข้อมูลย้อนหลังไม่พอสำหรับ Train Window (หุ้นบางตัวอาจเพิ่งเข้าตลาด)")
            st.stop()

        daily_ret = data.pct_change().dropna()
        voo_ret = daily_ret[benchmark]
        port_ret = daily_ret[my_portfolio]
        
        num_assets = len(my_portfolio)
        current_weights = np.zeros(num_assets)
        
        # เก็บผลลัพธ์
        strategy_returns = []
        bench_returns = []
        dates = []
        turnover_history = []
        
        status.update(label=f"🧮 กำลังรัน Optimizer แบบกลิ้งไปข้างหน้า (Rolling Window)...")
        
        # ------------------------------------------
        # 🔄 THE WALK-FORWARD LOOP
        # ------------------------------------------
        for i in range(train_window, len(port_ret), step_size):
            # 1. ข้อมูล Train (อดีต 252 วัน ณ จุดนั้น) -> ไม่มี Lookahead Bias
            train_data = port_ret.iloc[i - train_window : i]
            train_voo = voo_ret.iloc[i - train_window : i]
            
            # 2. คำนวณสัญญาณ Alpha (Residual Momentum 6 เดือน)
            mom_6m = (data[my_portfolio].iloc[i-1] / data[my_portfolio].iloc[i-126]) - 1
            voo_mom_6m = (data[benchmark].iloc[i-1] / data[benchmark].iloc[i-126]) - 1
            
            betas = []
            for t in my_portfolio:
                cov = np.cov(train_data[t], train_voo)[0][1]
                var = np.var(train_voo)
                betas.append(cov / var if var > 0 else 1.0)
            
            residual_mom = mom_6m.values - (np.array(betas) * voo_mom_6m)
            
            # ปรับเป็น Z-Score เพื่อใช้เป็น Q (Expected Return)
            if residual_mom.std() > 0: alpha_scores = (residual_mom - residual_mom.mean()) / residual_mom.std()
            else: alpha_scores = np.zeros(num_assets)
            
            # 3. Ledoit-Wolf Covariance
            lw = LedoitWolf()
            lw.fit(train_data)
            shrunk_cov = lw.covariance_ * 252
            
            # 4. Mean-Variance Optimization พร้อม Turnover Constraint
            def objective(w):
                port_ret_exp = np.sum(alpha_scores * w)
                port_var = np.dot(w.T, np.dot(shrunk_cov, w))
                # Maximize Return, Minimize Variance, Penalize Turnover
                turnover_penalty = 0.05 * np.sum(np.abs(w - current_weights))
                return -(port_ret_exp - (2.0 / 2.0) * port_var - turnover_penalty)
            
            bounds = tuple((0.0, 0.30) for _ in range(num_assets)) # Max weight 30%
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            try:
                res = minimize(objective, [1/num_assets]*num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
                target_weights = res.x if res.success else current_weights
            except:
                target_weights = current_weights
                
            # 5. หักต้นทุนการเทรด (Transaction Cost)
            turnover = np.sum(np.abs(target_weights - current_weights)) / 2.0 # หาร 2 เพราะ ซื้อ+ขาย คือ 1 ธุรกรรม
            turnover_history.append(turnover)
            cost_penalty = turnover * total_t_cost
            
            # 6. ข้อมูล Test (อนาคต 21 วัน ถัดไป)
            end_idx = min(i + step_size, len(port_ret))
            test_data = port_ret.iloc[i : end_idx]
            test_voo = voo_ret.iloc[i : end_idx]
            
            # คำนวณผลตอบแทนพอร์ตในช่วง Test
            step_returns = np.dot(test_data, target_weights)
            
            # หักค่าคอมฯ ออกจากวันแรกของการปรับพอร์ต
            if len(step_returns) > 0:
                step_returns[0] -= cost_penalty
            
            strategy_returns.extend(step_returns)
            bench_returns.extend(test_voo.values)
            dates.extend(test_data.index)
            
            # อัปเดตน้ำหนักสำหรับรอบถัดไป
            current_weights = target_weights

        status.update(label="--- การจำลองเสร็จสิ้น ---", state="complete")

        # ==========================================
        # 📊 การคำนวณและแสดงผล (Analytics)
        # ==========================================
        df_results = pd.DataFrame({'Strategy': strategy_returns, 'Benchmark': bench_returns}, index=dates)
        
        # คำนวณ Equity Curve (เริ่ม 100)
        df_cum = (1 + df_results).cumprod() * 100
        
        st.markdown(f"### 📈 เส้นโค้งความมั่งคั่งที่แท้จริง (Real Equity Curve)")
        fig_eq = px.line(df_cum, x=df_cum.index, y=['Strategy', 'Benchmark'], color_discrete_sequence=['#00d4ff', '#ff4b4b'])
        fig_eq.update_layout(yaxis_title="มูลค่าพอร์ต (เริ่ม 100)", legend_title="", height=400)
        st.plotly_chart(fig_eq, use_container_width=True)
        
        # คำนวณ Metrics ระดับสถาบัน
        years = len(df_results) / 252
        
        def calc_metrics(returns, cum_returns):
            cagr = ((cum_returns.iloc[-1] / 100) ** (1 / years)) - 1
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = cagr / ann_vol if ann_vol > 0 else 0
            
            roll_max = cum_returns.cummax()
            mdd = ((cum_returns - roll_max) / roll_max).min()
            return cagr*100, ann_vol*100, sharpe, mdd*100

        s_cagr, s_vol, s_sharpe, s_mdd = calc_metrics(df_results['Strategy'], df_cum['Strategy'])
        b_cagr, b_vol, b_sharpe, b_mdd = calc_metrics(df_results['Benchmark'], df_cum['Benchmark'])
        avg_turnover = (np.mean(turnover_history) * 12) * 100 # Annualized Turnover

        st.markdown("### 🏆 สรุปมาตรวัดระดับสถาบัน (Institutional Metrics)")
        metrics_df = pd.DataFrame({
            "Metric": ["CAGR (ผลตอบแทน/ปี)", "Volatility (ความผันผวน)", "Sharpe Ratio", "Max Drawdown", "Annual Turnover (อัตราหมุนเวียนพอร์ต)"],
            "Quant Strategy": [f"{s_cagr:.2f}%", f"{s_vol:.2f}%", f"{s_sharpe:.2f}", f"{s_mdd:.2f}%", f"{avg_turnover:.1f}%"],
            benchmark: [f"{b_cagr:.2f}%", f"{b_vol:.2f}%", f"{b_sharpe:.2f}", f"{b_mdd:.2f}%", "N/A"]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.info(f"💡 **Insights:** ระบบคำนวณหักต้นทุนค่าธรรมเนียมและ Slippage ไปทั้งหมดแล้ว นี่คือผลตอบแทนสุทธิ (Net Return) ที่ป้องกันการเกิด Lookahead Bias อย่างสมบูรณ์แบบ")
