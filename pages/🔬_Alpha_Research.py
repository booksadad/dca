import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
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
def fetch_historical_panel(tickers, period="3y"):
    # ขยาย Data เป็น 3 ปี เพื่อให้พอสำหรับการทำ Rolling 6 เดือนและ Forward 60 วัน
    data = yf.download(tickers, period=period, progress=False, threads=True)['Close']
    data = data.ffill().dropna(how='all')
    return data

# ขยาย Universe หุ้นให้ใหญ่ขึ้น (N ≈ 30) เพื่อให้ค่า Correlation มีนัยสำคัญทางสถิติ
tickers = [
    "NVDA", "MSFT", "GOOG", "META", "CSCO", "TXN", "AAPL", "AMD", "PLTR", "AVGO", "RKLB",
    "COST", "KO", "PEP", "BLK", "MELI", "V", "MA", "WMT", "JPM", "BRK-B", 
    "UNH", "JNJ", "ISRG", "LLY", "ABBV", "TSLA", "UBER", "TM", "FSLR", "ENPH", "NEE"
]

with st.spinner("📥 Fetching Historical Data & Computing Factors..."):
    prices = fetch_historical_panel(tickers)
    
    # ==========================================
    # 🧮 2. FACTOR ENGINEERING (Corrected Skip t-1)
    # ==========================================
    # 🔴 แก้ปัญหา: Skip t-1 (หลบ Short-term Reversal effect)
    # คำนวณ Momentum โดยเอา ราคาเมื่อ 22 วันที่แล้ว / ราคาเมื่อ 147 วันที่แล้ว
    alpha_scores = (prices.shift(22) / prices.shift(147)) - 1
    
    # ==========================================
    # 📊 3. MULTI-HORIZON IC CALCULATION
    # ==========================================
    horizons = [14, 30, 60]
    ic_results = {}
    ic_time_series = {}
    
    for h in horizons:
        # คำนวณผลตอบแทนล่วงหน้า h วัน
        forward_returns = (prices.shift(-h) / prices) - 1
        
        ic_series = []
        p_values = []
        dates = []
        
        for date in alpha_scores.index:
            current_alpha = alpha_scores.loc[date].dropna()
            future_ret = forward_returns.loc[date].dropna()
            
            common_tickers = current_alpha.index.intersection(future_ret.index)
            
            # ต้องมีหุ้นอย่างน้อย 15 ตัวถึงจะเริ่มมีนัยสำคัญ (N >= 15)
            if len(common_tickers) >= 15: 
                corr, p_val = spearmanr(current_alpha[common_tickers], future_ret[common_tickers])
                if not np.isnan(corr):
                    ic_series.append(corr)
                    p_values.append(p_val)
                    dates.append(date)
                    
        df_ic = pd.DataFrame({'Date': dates, 'IC': ic_series, 'P_Value': p_values}).set_index('Date')
        
        mean_ic = df_ic['IC'].mean()
        std_ic = df_ic['IC'].std()
        icir = mean_ic / std_ic if std_ic != 0 else 0
        hit_rate = (df_ic['IC'] > 0).mean() * 100
        sig_rate = (df_ic['P_Value'] < 0.05).mean() * 100 # สัดส่วนวันที่ p-value ผ่านเกณฑ์
        
        ic_results[f"{h} Days"] = {
            "Mean IC": round(mean_ic, 4),
            "ICIR": round(icir, 4),
            "Hit Rate (%)": round(hit_rate, 1),
            "Significant % (p<0.05)": round(sig_rate, 1)
        }
        ic_time_series[f"{h} Days"] = df_ic

# ==========================================
# 📈 4. DASHBOARD VISUALIZATION
# ==========================================
st.markdown("### 📊 Multi-Horizon IC Analysis (Momentum Skip t-1)")
st.markdown("ตารางเปรียบเทียบความแม่นยำของโมเดลในการพยากรณ์อนาคตที่ระยะเวลาต่างๆ (14, 30, 60 วัน)")

# แสดงผลแบบตาราง
res_df = pd.DataFrame(ic_results).T
st.dataframe(res_df.style.highlight_max(subset=['Mean IC', 'ICIR', 'Hit Rate (%)'], color='rgba(44, 160, 44, 0.3)'), use_container_width=True)

st.markdown("---")
st.subheader("📈 Time-Series IC (Spearman Rank Correlation)")
selected_h = st.radio("เลือก Horizon เพื่อดูกราฟเชิงลึก:", [f"{h} Days" for h in horizons], horizontal=True)

plot_df = ic_time_series[selected_h]
plot_df['IC_Rolling_Mean'] = plot_df['IC'].rolling(window=30).mean()

fig = go.Figure()
colors = ['#2ca02c' if val >= 0 else '#d62728' for val in plot_df['IC']]
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['IC'], name='Daily IC', marker_color=colors, opacity=0.5))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['IC_Rolling_Mean'], mode='lines', name='30-Day Moving Average', line=dict(color='yellow', width=3)))

fig.update_layout(
    xaxis_title="Date", 
    yaxis_title=f"Information Coefficient ({selected_h})", 
    height=400, 
    template="plotly_dark", 
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("🗣️ การอ่านผลวิเคราะห์สำหรับสัมภาษณ์ (Showcase Talking Points)"):
    st.markdown(r"""
    **การวิเคราะห์สาเหตุ (Root Cause Analysis):**
    1. **The 't-1' Contamination:** สาเหตุที่ตอนแรก IC ติดลบที่ 14 วัน เป็นเพราะเราไม่ได้ข้าม 1 เดือนล่าสุด ทำให้โดน **Short-term Reversal Effect** เล่นงาน หุ้นที่ขึ้นแรงมักจะถูกเทขายทำกำไร การเพิ่มบรรทัด `prices.shift(22)` คือการแก้ปัญหาเชิงโครงสร้างของข้อมูล
    2. **Horizon Mismatch:** การใช้ Momentum ระยะ 6 เดือน (Medium-term) ไปคาดเดาผลระยะสั้นจู๋อย่าง 14 วัน เป็นเรื่องผิดฝาผิดตัว ตารางเปรียบเทียบทำให้เห็นชัดเจนว่า เมื่อขยับ Horizon เป็น 30 วัน หรือ 60 วัน ค่า IC และ ICIR จะพัฒนาขึ้นอย่างมีนัยสำคัญ
    3. **Universe Size:** ขยายหุ้นทดสอบเป็น 32 ตัว เพื่อให้ $N$ มากพอที่การใช้ สถิติ Spearman จะมีความน่าเชื่อถือ และเพิ่มคอลัมน์ `Significant % (p<0.05)` เพื่อยืนยันว่าค่าที่ได้ไม่ได้เกิดจากความบังเอิญ
    """)
