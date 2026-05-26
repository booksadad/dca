import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings("ignore")

PORTFOLIO_FILE = "my_portfolio_data.csv"

st.set_page_config(page_title="QuantHQ Tracker", page_icon="📈", layout="wide")
st.title("📈 QUANT-HQ: Strategy Lab & Portfolio Analytics")
st.markdown("ระบบจำลองผลตอบแทนย้อนหลัง (Backtesting) และตรวจสอบมาตรวัดระดับสถาบัน")
st.markdown("---")

# 1. โหลดข้อมูลพอร์ตปัจจุบัน
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

st.sidebar.markdown(f"**💼 หุ้นในพอร์ตปัจจุบัน:**\n{', '.join(my_portfolio)}")

# 2. ตั้งค่า Backtest
col1, col2 = st.columns(2)
with col1:
    lookback_years = st.slider("🕰️ ระยะเวลาย้อนหลัง (ปี)", 1, 10, 3)
with col2:
    benchmark_ticker = st.selectbox("📊 ดัชนีเปรียบเทียบ (Benchmark)", ["VOO", "QQQ", "DIA"])

if st.button("🚀 รันระบบ Backtest & Analytics", type="primary"):
    with st.spinner(f"⏳ กำลังดึงข้อมูลย้อนหลัง {lookback_years} ปี และจำลอง Equity Curve..."):
        
        # ดึงข้อมูลราคา
        tickers_to_fetch = list(set(my_portfolio + [benchmark_ticker]))
        data = yf.download(tickers_to_fetch, period=f"{lookback_years}y", progress=False)['Close']
        data = data.dropna()
        
        if data.empty:
            st.error("❌ ดึงข้อมูลล้มเหลว หรือหุ้นบางตัวเพิ่งเข้าตลาดไม่ถึงเวลาที่กำหนด")
            st.stop()

        # 3. คำนวณผลตอบแทนรายวัน (Daily Returns) แบบ Equal Weight
        daily_ret = data.pct_change().dropna()
        
        valid_port_tickers = [t for t in my_portfolio if t in daily_ret.columns]
        port_weight = 1.0 / len(valid_port_tickers)
        
        # สมมติฐานพอร์ต: Equal Weight Rebalance ทุกวัน
        port_daily_ret = (daily_ret[valid_port_tickers] * port_weight).sum(axis=1)
        bench_daily_ret = daily_ret[benchmark_ticker]
        
        # คำนวณ Equity Curve (เส้นความมั่งคั่ง) เริ่มต้น 100
        port_cum = (1 + port_daily_ret).cumprod() * 100
        bench_cum = (1 + bench_daily_ret).cumprod() * 100
        
        # 4. คำนวณ Institutional Metrics
        years = len(daily_ret) / 252
        
        def calc_metrics(daily_returns, cum_returns):
            cagr = ((cum_returns.iloc[-1] / 100) ** (1 / years)) - 1
            ann_vol = daily_returns.std() * np.sqrt(252)
            sharpe = cagr / ann_vol if ann_vol > 0 else 0
            
            # Sortino Ratio (ลงโทษเฉพาะขาลง)
            downside = daily_returns.copy()
            downside[downside > 0] = 0
            down_vol = downside.std() * np.sqrt(252)
            sortino = cagr / down_vol if down_vol > 0 else 0
            
            # Max Drawdown
            roll_max = cum_returns.cummax()
            drawdown = (cum_returns - roll_max) / roll_max
            mdd = drawdown.min()
            
            return cagr*100, ann_vol*100, sharpe, sortino, mdd*100

        port_cagr, port_vol, port_sharpe, port_sortino, port_mdd = calc_metrics(port_daily_ret, port_cum)
        bench_cagr, bench_vol, bench_sharpe, bench_sortino, bench_mdd = calc_metrics(bench_daily_ret, bench_cum)

        # ==========================================
        # 📊 แสดงผล (Visualizations)
        # ==========================================
        st.markdown(f"### 📈 เส้นโค้งความมั่งคั่ง (Equity Curve) เทียบ {benchmark_ticker}")
        
        df_plot = pd.DataFrame({'วันที่': port_cum.index, 'Quant Portfolio': port_cum.values, 'Benchmark': bench_cum.values})
        fig_eq = px.line(df_plot, x='วันที่', y=['Quant Portfolio', 'Benchmark'], 
                         color_discrete_sequence=['#00d4ff', '#ff4b4b'])
        fig_eq.update_layout(yaxis_title="มูลค่าพอร์ต (เริ่ม 100)", legend_title="", height=400)
        st.plotly_chart(fig_eq, use_container_width=True)
        
        st.markdown("### 🏆 สรุปมาตรวัดระดับสถาบัน (Institutional Metrics)")
        
        # สร้างตารางเปรียบเทียบ
        metrics_df = pd.DataFrame({
            "Metric": ["CAGR (ผลตอบแทน/ปี)", "Volatility (ความผันผวน)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (หลุมยุบ)"],
            "Quant Portfolio": [f"{port_cagr:.2f}%", f"{port_vol:.2f}%", f"{port_sharpe:.2f}", f"{port_sortino:.2f}", f"{port_mdd:.2f}%"],
            benchmark_ticker: [f"{bench_cagr:.2f}%", f"{bench_vol:.2f}%", f"{bench_sharpe:.2f}", f"{bench_sortino:.2f}", f"{bench_mdd:.2f}%"]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # ------------------------------------------
        # ⚠️ DRAWDOWN MONITOR
        # ------------------------------------------
        st.markdown("---")
        st.markdown("### 🕳️ แผนผังความเจ็บปวด (Drawdown Monitor)")
        
        port_roll_max = port_cum.cummax()
        port_dd = (port_cum - port_roll_max) / port_roll_max * 100
        
        fig_dd = px.area(x=port_dd.index, y=port_dd.values, color_discrete_sequence=['#ff4b4b'])
        fig_dd.update_layout(xaxis_title="วันที่", yaxis_title="Drawdown (%)", height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # ------------------------------------------
        # 📡 RISK EXPOSURE ANALYSIS
        # ------------------------------------------
        st.markdown("### 📡 วิเคราะห์ความเสี่ยงรายตัว (Risk Exposure)")
        
        risk_data = []
        for t in valid_port_tickers:
            t_ret = daily_ret[t]
            t_cum = (1 + t_ret).cumprod() * 100
            t_cagr, t_vol, t_sharpe, _, t_mdd = calc_metrics(t_ret, t_cum)
            
            # คำนวณ Beta (ความแกว่งเทียบตลาด)
            cov = np.cov(t_ret, bench_daily_ret)[0][1]
            var = np.var(bench_daily_ret)
            beta = cov / var if var > 0 else 1.0
            
            risk_data.append({
                "หุ้น": t, "CAGR (%)": t_cagr, "Volatility (%)": t_vol, 
                "Sharpe": t_sharpe, "Max Drawdown (%)": t_mdd, "Beta (vs Mkt)": beta
            })
            
        # 🛠️ ปิดบั๊กตาราง: ตัด style.background_gradient ออก ใช้ตารางธรรมดาที่เสถียร 100%
        df_risk = pd.DataFrame(risk_data).round(2)
        st.dataframe(df_risk, use_container_width=True, hide_index=True)
