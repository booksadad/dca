import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go # 🌟 นำเข้าของเล่นใหม่ กราฟระดับโปร!

st.set_page_config(page_title="QuantHQ | Macro Radar", page_icon="🌍", layout="wide")

st.title("🌍 Macroeconomics & Market Weather Radar (Pro Edition)")
st.markdown("หอสังเกตการณ์เศรษฐกิจโลก วิเคราะห์ดัชนีความกลัวและทิศทางตลาด ด้วยกราฟแบบ Interactive")
st.markdown("---")

with st.spinner("📡 กำลังเชื่อมต่อดาวเทียม ดึงข้อมูลจาก Wall Street..."):
    try:
        vix_ticker, voo_ticker, tnx_ticker = "^VIX", "VOO", "^TNX"
        data = yf.download([vix_ticker, voo_ticker, tnx_ticker], period="1y", progress=False)['Close']
        
        current_vix = data[vix_ticker].dropna().iloc[-1]
        prev_vix = data[vix_ticker].dropna().iloc[-2]
        current_voo = data[voo_ticker].dropna().iloc[-1]
        sma200_voo = data[voo_ticker].dropna().rolling(200).mean().iloc[-1]
        current_tnx = data[tnx_ticker].dropna().iloc[-1]
        prev_tnx = data[tnx_ticker].dropna().iloc[-2]

        # ==========================================
        # 🧠 ระบบ AI แนะนำสมองกล (Engine Recommendation)
        # ==========================================
        c_meter, c_text = st.columns([1, 2])
        
        if current_vix > 25 or current_voo < sma200_voo:
            market_state, rec_engine, alert_color = "🔴 พายุเข้า / ตลาดผันผวนสูง", "🛡️ Risk Parity", "error"
            reason = "VIX พุ่งสูง หรือ ตลาดหลุดเส้น SMA 200 วัน แนะนำให้หลบภัยในหุ้นปันผลและหุ้น Defensive"
        elif current_vix > 18:
            market_state, rec_engine, alert_color = "🟡 ตลาดเฝ้าระวัง (Watchlist)", "🛡️ Risk Parity สลับกับ 🚀 Alpha", "warning"
            reason = "ความตื่นตระหนกเริ่มก่อตัว แนะนำให้จัดพอร์ตแบบผสมผสาน ไม่ควรทุ่มหุ้นซิ่ง 100%"
        else:
            market_state, rec_engine, alert_color = "🟢 ท้องฟ้าสดใส (Bull Market)", "🚀 Alpha Momentum", "success"
            reason = "ความกลัวต่ำ ตลาดขาขึ้นชัดเจน เป็นจังหวะดีที่จะกอบโกยกำไรจากหุ้น Growth"

        with c_text:
            st.subheader("🤖 AI สรุปสภาวะตลาดประจำสัปดาห์")
            if alert_color == "error": st.error(f"**สภาพอากาศ:** {market_state}\n\n**สมองกลที่แนะนำ:** {rec_engine}\n\n*เหตุผล:* {reason}")
            elif alert_color == "warning": st.warning(f"**สภาพอากาศ:** {market_state}\n\n**สมองกลที่แนะนำ:** {rec_engine}\n\n*เหตุผล:* {reason}")
            else: st.success(f"**สภาพอากาศ:** {market_state}\n\n**สมองกลที่แนะนำ:** {rec_engine}\n\n*เหตุผล:* {reason}")

        # 🌟 เพิ่มหน้าปัด (Gauge Meter) ให้ดูโปรสุดๆ
        with c_meter:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_vix,
                title = {'text': "🌡️ ดัชนีความกลัว (VIX Meter)"},
                gauge = {
                    'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "black"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 18], 'color': "#a8f0c6"}, # เขียว
                        {'range': [18, 25], 'color': "#fce988"}, # เหลือง
                        {'range': [25, 40], 'color': "#ffb3b3"}], # แดง
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': current_vix}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # ==========================================
        # 📊 หน้าปัดดัชนี (Dashboard - Plotly Edition)
        # ==========================================
        c1, c2, c3 = st.columns(3)
        
        # 1. VIX Index 
        with c1:
            st.markdown("### 😨 VIX Index")
            st.metric("VIX Level", f"{current_vix:.2f}", f"{current_vix - prev_vix:.2f}", delta_color="inverse")
            
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(x=data.index[-90:], y=data[vix_ticker].tail(90), mode='lines', name='VIX', line=dict(color='red', width=2)))
            # ขีดเส้นเตือนภัย
            fig_vix.add_hline(y=18, line_dash="dot", line_color="orange", annotation_text=" เฝ้าระวัง (18)", annotation_position="bottom right")
            fig_vix.add_hline(y=25, line_dash="dash", line_color="red", annotation_text=" อันตราย (25)", annotation_position="top right")
            fig_vix.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_vix, use_container_width=True)
            
        # 2. S&P 500 (VOO)
        with c2:
            st.markdown("### 📈 S&P 500 (VOO)")
            st.metric("Price vs SMA200", f"${current_voo:.2f}", f"{(current_voo - sma200_voo):.2f} (เหนือเส้น 200 วัน)")
            
            fig_voo = go.Figure()
            fig_voo.add_trace(go.Scatter(x=data.index[-90:], y=data[voo_ticker].tail(90), mode='lines', name='Price', line=dict(color='#2E86C1', width=2)))
            fig_voo.add_trace(go.Scatter(x=data.index[-90:], y=data[voo_ticker].rolling(200).mean().tail(90), mode='lines', name='SMA 200', line=dict(color='#F39C12', width=2, dash='dash')))
            fig_voo.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_voo, use_container_width=True)
            
        # 3. US 10-Year Treasury Yield
        with c3:
            st.markdown("### 💵 US 10-Yr Yield")
            st.metric("10-Year Yield", f"{current_tnx:.2f}%", f"{(current_tnx - prev_tnx):.2f}%", delta_color="inverse")
            
            fig_tnx = go.Figure()
            # ทำกราฟแบบระบายสีใต้เส้นให้ดูแพง
            fig_tnx.add_trace(go.Scatter(x=data.index[-90:], y=data[tnx_ticker].tail(90), fill='tozeroy', mode='lines', name='Yield', line=dict(color='#27AE60', width=2)))
            fig_tnx.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_tnx, use_container_width=True)

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูลจากดาวเทียม: {str(e)}")
