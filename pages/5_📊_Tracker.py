import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="The 400M Tracker", page_icon="📈", layout="wide")

# ==========================================
# 💾 ดึงข้อมูลพอร์ตปัจจุบันอัตโนมัติ
# ==========================================
PORTFOLIO_FILE = "my_portfolio_data.csv"
current_port_value = 0.0

if os.path.exists(PORTFOLIO_FILE):
    try:
        saved_df = pd.read_csv(PORTFOLIO_FILE)
        if not saved_df.empty and "ยอดเงินปัจจุบัน (บาท)" in saved_df.columns:
            current_port_value = pd.to_numeric(saved_df["ยอดเงินปัจจุบัน (บาท)"], errors='coerce').sum()
    except Exception:
        pass

# ==========================================
# 🎨 UI & Header
# ==========================================
st.title("📈 THE 400M TRACKER (สมุดพกความรวย)")
st.markdown("ระบบคำนวณและติดตามเส้นทางสู่อิสรภาพทางการเงิน **เป้าหมาย: 400 ล้านบาท** 🚀")
st.markdown("---")

# ==========================================
# 🎛️ แผงควบคุม (Inputs)
# ==========================================
st.sidebar.subheader("⚙️ ตัวแปรสมการความมั่งคั่ง")
current_age = st.sidebar.number_input("อายุปัจจุบัน (ปี)", min_value=15, max_value=80, value=21)
target_age = st.sidebar.number_input("เป้าหมายอายุที่อยากเกษียณ (ปี)", min_value=30, max_value=100, value=40)

st.sidebar.markdown("---")
start_principal = st.sidebar.number_input("เงินต้นปัจจุบันในพอร์ต (บาท)", min_value=0.0, value=float(current_port_value), step=1000.0)
monthly_dca = st.sidebar.number_input("เงินเติม DCA รายเดือน (บาท)", min_value=0.0, value=4000.0, step=500.0)
st.sidebar.caption("💡 แนะนำ: 4,000 บ./เดือน (เท่ากับสัปดาห์ละ 1,000 บ.)")

st.sidebar.markdown("---")
expected_cagr = st.sidebar.slider("ผลตอบแทนคาดหวังต่อปี (CAGR %)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
st.sidebar.caption("🎯 เป้าหมายระบบ Max Sharpe ของเราคือ 15% - 20% ต่อปี")

# ==========================================
# 🧮 คณิตศาสตร์ประยุกต์: Compound Interest
# ==========================================
years_to_grow = target_age - current_age
months_to_grow = years_to_grow * 12

# แปลงผลตอบแทนรายปี เป็นรายเดือน
monthly_rate = (1 + expected_cagr / 100) ** (1/12) - 1

ages = []
portfolio_values = []
total_invested_list = []

current_balance = start_principal
total_invested = start_principal

for month in range(months_to_grow + 1):
    if month % 12 == 0:
        ages.append(current_age + (month // 12))
        portfolio_values.append(current_balance)
        total_invested_list.append(total_invested)
        
    # ดอกเบี้ยทบต้น + เติมเงินรายเดือน
    current_balance = current_balance * (1 + monthly_rate) + monthly_dca
    total_invested += monthly_dca

final_value = portfolio_values[-1]
target_goal = 400_000_000
progress_percent = min((final_value / target_goal) * 100, 100)

# ==========================================
# 📊 แดชบอร์ดแสดงผล (Metrics)
# ==========================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("💼 ยอดเงินปัจจุบัน", f"฿{start_principal:,.2f}")
c2.metric(f"🎯 คาดการณ์ตอนอายุ {target_age} ปี", f"฿{final_value:,.2f}", f"{progress_percent:.2f}% ของเป้า 400M")
c3.metric("💸 เงินต้นที่ควักกระเป๋าจริง", f"฿{total_invested_list[-1]:,.2f}")
c4.metric("✨ ดอกเบี้ย/กำไรที่ระบบทำได้", f"฿{(final_value - total_invested_list[-1]):,.2f}")

if final_value >= target_goal:
    st.success(f"🎉 **ยินดีด้วย!** ด้วยแผนนี้ คุณจะบรรลุเป้าหมาย 400 ล้านบาท ได้ทันเวลาอย่างแน่นอน!")
else:
    st.warning(f"⚠️ **Mission Alert:** เป้าหมายยังขาดอีก ฿{(target_goal - final_value):,.2f} ลองปรับเพิ่มเงิน DCA หรือยืดอายุเกษียณดูครับ")

# ==========================================
# 📈 กราฟเส้นทางความมั่งคั่ง (Wealth Trajectory)
# ==========================================
st.markdown("### 🌌 แผนที่เดินทาง (Wealth Trajectory)")

fig = go.Figure()

# เส้นเป้าหมาย 400 ล้าน
fig.add_hline(y=target_goal, line_dash="dash", line_color="red", annotation_text="เป้าหมาย 400 ล้านบาท", annotation_position="top left")

# พื้นที่เงินต้นที่เติมเข้าไป (ทุน)
fig.add_trace(go.Scatter(
    x=ages, y=total_invested_list,
    mode='lines',
    fill='tozeroy',
    name='เงินต้น (Principal)',
    line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
))

# พื้นที่พอร์ตโฟลิโอรวม (เงินต้น + กำไร)
fig.add_trace(go.Scatter(
    x=ages, y=portfolio_values,
    mode='lines',
    fill='tonexty',
    name='มูลค่าพอร์ต (Portfolio Value)',
    line=dict(color='#00ff88', width=4)
))

fig.update_layout(
    xaxis_title="อายุ (Age)",
    yaxis_title="มูลค่าพอร์ต (THB)",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🏆 ระบบแจ้งเตือนจุดเช็คพอยต์ (Milestones)
# ==========================================
st.markdown("### 🏆 จุดเช็คพอยต์ความมั่งคั่ง (Milestones)")

milestones = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 400_000_000]
achieved_ages = {}

for m in milestones:
    for age, val in zip(ages, portfolio_values):
        if val >= m:
            achieved_ages[m] = age
            break

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
milestone_cols = [col_m1, col_m2, col_m3, col_m4, col_m5]
milestone_labels = ["1 ล้านบาท", "10 ล้านบาท", "50 ล้านบาท", "100 ล้านบาท", "400 ล้านบาท"]

for idx, m in enumerate(milestones):
    with milestone_cols[idx]:
        if m in achieved_ages:
            st.success(f"**{milestone_labels[idx]}**\n\nบรรลุตอนอายุ: **{achieved_ages[m]} ปี** 🟢")
        else:
            st.error(f"**{milestone_labels[idx]}**\n\nยังไปไม่ถึงในกรอบเวลา 🔴")

# ==========================================
# 🥊 เปรียบเทียบกับตลาดโลก (S&P 500 Benchmark)
# ==========================================
st.markdown("---")
with st.expander("🥊 เปรียบเทียบผลงาน: พอร์ตของเรา VS ตลาดโลก (S&P 500)", expanded=False):
    st.markdown("สมมติฐาน: S&P 500 ให้ผลตอบแทนเฉลี่ยระยะยาวที่ **10% ต่อปี** (CAGR)")
    
    sp500_rate = (1 + 10.0 / 100) ** (1/12) - 1
    sp500_values = []
    current_sp500 = start_principal
    
    for month in range(months_to_grow + 1):
        if month % 12 == 0:
            sp500_values.append(current_sp500)
        current_sp500 = current_sp500 * (1 + sp500_rate) + monthly_dca
        
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Scatter(
        x=ages, y=portfolio_values,
        mode='lines', name=f'พอร์ตเรา ({expected_cagr}% CAGR)',
        line=dict(color='#00ff88', width=3)
    ))
    
    fig_comp.add_trace(go.Scatter(
        x=ages, y=sp500_values,
        mode='lines', name='ซื้อทิ้ง S&P 500 (10% CAGR)',
        line=dict(color='#ff4b4b', width=2, dash='dash')
    ))
    
    fig_comp.update_layout(
        xaxis_title="อายุ (Age)", yaxis_title="มูลค่าพอร์ต (THB)",
        height=400, hovermode="x unified"
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    diff = final_value - sp500_values[-1]
    st.info(f"💡 การจัดพอร์ตแบบ Quant ด้วยระบบของเรา สร้างส่วนต่างกำไรเหนือตลาดโลกได้ถึง **฿{diff:,.2f}** ในระยะเวลา {years_to_grow} ปี!")
