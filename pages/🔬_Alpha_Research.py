import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Alpha Research Lab", page_icon="🔬", layout="wide")
st.title("🔬 Alpha Research Lab (IC / ICIR Framework)")
st.markdown("ห้องแล็บวิจัยสำหรับวัดประสิทธิภาพและพลังการพยากรณ์ (Predictive Power) ของ Alpha Factor ด้วยสถิติ **Spearman Rank Correlation**")

# ==========================================
# ⚙️ 1. SETUP & DATA FETCHING
# ==========================================
@st.cache_data(ttl=86400)
def fetch_historical_panel(tickers, period="2y"):
    data = yf.download(tickers, period=period, progress=False)['Close']
    data = data.ffill().dropna(how='all')
    return data

tickers = ["NVDA", "MSFT", "GOOG", "META", "AAPL", "TSLA", "JNJ", "KO", "WMT", "JPM", "BRK-B", "ENPH"]
st.sidebar.header("⚙️ Research Parameters")
forward_days = st.sidebar.slider("Forward Return Horizon (Days)", 5, 60, 14, 5)

with st.spinner("📥 Fetching Historical Data & Computing Factors..."):
    prices = fetch_historical_panel(tickers)
    
    # ==========================================
    # 🧮 2. FACTOR ENGINEERING (Historical Vectorized)
    # ==========================================
    # 1. Forward Returns (ผลตอบแทนในอนาคต N วัน เพื่อเอามาเป็นตัวเฉลย)
    forward_returns = prices.shift(-forward_days) / prices - 1
    
    # 2. Historical Momentum (Skip t-1 month / 22 days)
    # คำนวณผลตอบแทนย้อนหลัง 6 เดือน โดยเว้น 1 เดือนล่าสุด
    past_147 = prices.shift(22)
    past_126 = prices.shift(147)
    momentum_factor = (past_147 / past_126) - 1
    
    # 3. Simple Mean Reversion (1M Return - หุ้นที่ขึ้นแรงระยะสั้นมักจะย่อ)
    short_term_rev = (prices / prices.shift(22)) - 1
    
    # รวม Factor แบบจำลอง (Mock Alpha Score) สำหรับ Backtest
    # (ใน Research จริงจะใช้ Factor พื้นฐานด้วย แต่นี่คือการจำลอง Time-series อย่างรวดเร็ว)
    alpha_scores = momentum_factor - (short_term_rev * 0.5)

    # ==========================================
    # 📊 3. IC / ICIR CALCULATION
    # ==========================================
    ic_series = []
    dates = []
    
    # คำนวณ Cross-sectional Rank Correlation วันต่อวัน
    for date in alpha_scores.index:
        current_alpha = alpha_scores.loc[date].dropna()
        future_ret = forward_returns.loc[date].dropna()
        
        # หาหุ้นที่มีข้อมูลครบทั้ง Alpha และ Forward Return ในวันนั้น
        common_tickers = current_alpha.index.intersection(future_ret.index)
        
        if len(common_tickers) >= 5: # ต้องมีหุ้นอย่างน้อย 5 ตัวถึงจะจัด Rank ได้
            corr, _ = spearmanr(current_alpha[common_tickers], future_ret[common_tickers])
            if not np.isnan(corr):
                ic_series.append(corr)
                dates.append(date)

    ic_df = pd.DataFrame({'Date': dates, 'IC': ic_series}).set_index('Date')
    ic_df['IC_Rolling_Mean'] = ic_df['IC'].rolling(window=30).mean() # เส้นค่าเฉลี่ย 1 เดือน

    # สถิติสรุป
    mean_ic = ic_df['IC'].mean()
    std_ic = ic_df['IC'].std()
    icir = mean_ic / std_ic if std_ic != 0 else 0
    hit_rate = (ic_df['IC'] > 0).mean() * 100

# ==========================================
# 📈 4. DASHBOARD VISUALIZATION
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Information Coefficient (Mean IC)", f"{mean_ic:.4f}", help="> 0.05 คือดีมาก")
col2.metric("Information Ratio (ICIR)", f"{icir:.4f}", help="ความสม่ำเสมอของโมเดล > 0.5 คือเยี่ยม")
col3.metric("Signal Hit Rate", f"{hit_rate:.1f}%", help="สัดส่วนวันที่ความแม่นยำเป็นบวก")
col4.metric("Evaluation Horizon", f"{forward_days} Days", help="ระยะเวลาพยากรณ์ล่วงหน้า")

st.markdown("---")
st.subheader("📈 Time-Series IC (Spearman Rank Correlation)")
st.markdown(f"กราฟแสดงความสัมพันธ์ระหว่าง **Alpha Score** กับ **ผลตอบแทนอีก {forward_days} วันข้างหน้า**")

fig = go.Figure()
# กราฟแท่งแสดง IC รายวัน (สีเขียว=แม่นยำ, สีแดง=พยากรณ์ผิดพลาด)
colors = ['#2ca02c' if val >= 0 else '#d62728' for val in ic_df['IC']]
fig.add_trace(go.Bar(x=ic_df.index, y=ic_df['IC'], name='Daily IC', marker_color=colors, opacity=0.5))

# เส้นค่าเฉลี่ย Rolling
fig.add_trace(go.Scatter(x=ic_df.index, y=ic_df['IC_Rolling_Mean'], mode='lines', name='30-Day Moving Average', line=dict(color='yellow', width=3)))

fig.update_layout(xaxis_title="Date", yaxis_title="Information Coefficient (IC)", height=450, template="plotly_dark", showlegend=True)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📝 วิธีการตีความสำหรับสัมภาษณ์ (Interview Talking Points)"):
    st.markdown(r"""
    **คำอธิบายเชิงคณิตศาสตร์:**
    * ระบบนี้ใช้ **Spearman Rank Correlation** $(\rho)$ เพื่อวัดว่า ลำดับ (Rank) ของคะแนน Alpha ที่เราประเมินไว้ ตรงกับ ลำดับของผลตอบแทนที่เกิดขึ้นจริงในอนาคตหรือไม่
    * สูตร $IC = \rho(Rank_{Alpha}, Rank_{Forward\_Return})$ ช่วยตัดปัญหา Outlier ของราคาหุ้น ทำให้การวัดผลทนทาน (Robust) ขึ้น
    * **ICIR** ช่วยบอกว่าเราได้ Alpha มาแบบเสี่ยงดวงหรือมีระบบ ถ้า $ICIR > 0.5$ หมายความว่าโมเดลเราสร้างผลตอบแทนได้เสถียรมาก นำไปรันกองทุนได้จริง
    """)
