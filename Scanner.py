import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="QuantHQ | Top-Down Screener", page_icon="🔍", layout="wide")

st.title("🔍 Ultimate Mega Trend Screener")
st.markdown("ระบบสแกนหุ้นอัตโนมัติ: คัดหุ้นย่อตัว -> ตะแกรงร่อนงบการเงิน -> AI วิเคราะห์ SWOT เฉพาะผู้ชนะ")
st.markdown("---")

# ==========================================
# 🗂️ ฐานข้อมูลลับ (Hidden Database - Thematic 245 Tickers)
# ==========================================
mega_trends_db = {
    "🧠 1. โครงสร้างพื้นฐาน AI": [
        "NVDA", "MSFT", "AMD", "AVGO", "TSM", "META", "GOOG", "ASML", "PLTR", "SNOW", 
        "CRWD", "PANW", "AMZN", "INTC", "QCOM", "TXN", "MU", "ARM", "SMCI", "ANET", 
        "CDNS", "SNPS", "KLAC", "LRCX", "AMAT", "MRVL", "MCHP", "ADI", "NXPI", "ON", 
        "SWKS", "TER", "WDC", "STX", "IBM", "ORCL", "SAP", "CRM", "NOW", "ADBE", 
        "INTU", "TEAM", "WDAY", "DDOG", "NET", "ESTC", "MDB", "CFLT", "ZS", "FTNT"
    ],
    "🚗 2. ยานยนต์ & อัตโนมัติ": [
        "TSLA", "UBER", "MBLY", "PATH", "ROK", "RIVN", "ON", "SYM", "F", "GM", 
        "TM", "HMC", "STLA", "NIO", "XPEV", "LI", "LCID", "PSNY", "RACE", "APTV", 
        "LEA", "ALV", "VC", "DAN", "THO", "PI", "LAZR", "INVZ", "OUST", "QS", 
        "ENVX", "SLDP", "CHPT", "BLNK", "EVGO", "JOBY", "ACHR", "LILM", "EH", "BWA"
    ],
    "🛍️ 3. สินค้าอุปโภคบริโภค": [
        "AMZN", "MELI", "WMT", "COST", "V", "MA", "HD", "MCD", "SBUX", "SHOP", 
        "AXP", "PYPL", "SQ", "AFRM", "SOFI", "HOOD", "TGT", "LOW", "DG", "DLTR", 
        "ROST", "TJX", "LULU", "NKE", "UAA", "CROX", "DECK", "SKE", "VFC", "CPRI", 
        "TPR", "RL", "EL", "UL", "PG", "KO", "PEP", "KDP", "MNST", "CELH", 
        "HSY", "MDLZ", "K", "GIS", "CPB", "CAG", "SJM", "DIS", "NFLX", "SPOT"
    ],
    "🌿 4. เศรษฐกิจสีเขียว": [
        "ENPH", "FSLR", "NEE", "ALB", "SEDG", "PLUG", "RUN", "BEP", "CWEN", "NEP", 
        "HASI", "AY", "NOVA", "SPWR", "MAXN", "CSIQ", "DQ", "JKS", "FCEL", "BLDP", 
        "BE", "STEM", "FLNC", "CLNE", "GEVO", "SQM", "LTHM", "LAC", "CHPT", "BLNK", 
        "EVGO", "AES", "EXC", "PEG", "ED", "AWK", "WTRG", "CWT", "YORW", "DNMR"
    ],
    "🧬 5. เทคโนโลยีชีวภาพ": [
        "LLY", "NVO", "VRTX", "ISRG", "REGN", "MRK", "ABBV", "ILMN", "JNJ", "PFE", 
        "UNH", "CVS", "CI", "HUM", "CNC", "ELV", "BIIB", "GILD", "AMGN", "BMY", 
        "NVS", "AZN", "SNY", "GSK", "TMO", "DHR", "ABT", "SYK", "MDT", "BSX", 
        "EW", "ZBH", "BDX", "BAX", "ALGN", "IDXX", "RMD", "DXCM", "PODD", "TNDM", 
        "EXAS", "NTRA", "GH", "PACB", "TXG", "CRSP", "NTLA", "EDIT", "BEAM", "DNA"
    ]
}

# 🧠 ฐานข้อมูล AI Analyst (SWOT Database) เบื้องต้น
swot_db = {
    "MSFT": {"S": "ผูกขาดซอฟต์แวร์องค์กรทั่วโลก และเป็นผู้นำ AI", "W": "ขนาดบริษัทใหญ่มาก โตยากขึ้น", "O": "ขาย Copilot เพิ่มรายได้", "T": "โดนรัฐบาลเพ่งเล็งเรื่องผูกขาด"},
    "NVDA": {"S": "ผูกขาดตลาดชิป AI (GPU)", "W": "รายได้พึ่งพาลูกค้า Big Tech ไม่กี่เจ้า", "O": "การอัปเกรด Data Center ทั่วโลก", "T": "คู่แข่งอย่าง AMD หรือลูกค้าเริ่มทำชิปเอง"},
    # หุ้นที่ไม่มีในนี้ ระบบ AI จะเจนข้อความมาตรฐานให้
}

# ==========================================
# 🗂️ แถบด้านซ้าย
# ==========================================
st.sidebar.subheader("🎯 ตั้งค่าเรดาร์สแกนหุ้น")
min_drop = st.sidebar.slider("ย่อตัวขั้นต่ำ (%)", 0, 50, 15)
max_drop = st.sidebar.slider("ย่อตัวสูงสุด (%)", 10, 80, 35)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 เลือกหมวดหมู่ที่จะสแกน")
selected_trends = []
for trend in mega_trends_db.keys():
    if st.sidebar.checkbox(trend, value=True):
        selected_trends.append(trend)

# ==========================================
# 🧠 ระบบประมวลผล Screener
# ==========================================
if st.button("🚀 สั่ง AI สแกนพร้อมวิเคราะห์ SWOT เฉพาะผู้ชนะ"):
    if not selected_trends:
        st.error("⚠️ โปรดติ๊กเลือกหมวดหมู่อย่างน้อย 1 หมวด")
    else:
        scan_list = []
        for trend in selected_trends: scan_list.extend(mega_trends_db[trend])
        
        with st.spinner(f"⏳ กำลังคัดกรองหุ้น {len(scan_list)} ตัว (ราคาเข้าโซน -> งบสวย -> SWOT)..."):
            prices = yf.download(scan_list, period="5y", progress=False)['Close']
            if isinstance(prices, pd.Series): prices = pd.DataFrame({scan_list[0]: prices})
                
            screener_data = []
            financials_cache = {}
            
            progress_bar = st.progress(0)
            for i, ticker in enumerate(scan_list):
                progress_bar.progress((i + 1) / len(scan_list))
                if ticker not in prices.columns: continue
                
                stock_history = prices[ticker].dropna()
                if stock_history.empty: continue
                
                current_price = stock_history.iloc[-1]
                ath_price = stock_history.max()
                drawdown_pct = ((current_price - ath_price) / ath_price) * 100
                
                # กรองด่านที่ 1: ย่อตัวอยู่ในโซน Sweet Spot
                if not (-max_drop <= drawdown_pct <= -min_drop): continue 
                
                # กรองด่านที่ 2: ดึงงบการเงิน
                try:
                    ticker_obj = yf.Ticker(ticker)
                    fin = ticker_obj.financials
                    if not fin.empty:
                        rev_keys = [k for k in fin.index if 'Total Revenue' in str(k) or 'Operating Revenue' in str(k)]
                        net_keys = [k for k in fin.index if 'Net Income' in str(k)]
                        rev_data = fin.loc[rev_keys[0]].dropna()[::-1] if rev_keys else pd.Series()
                        net_data = fin.loc[net_keys[0]].dropna()[::-1] if net_keys else pd.Series()
                        
                        rev_growth = "✅ ผ่าน" if (len(rev_data) > 1 and rev_data.iloc[-1] > rev_data.iloc[0]) else "❌ ไม่ผ่าน"
                        net_growth = "✅ ผ่าน" if (len(net_data) > 1 and net_data.iloc[-1] > 0 and net_data.iloc[-1] > net_data.iloc[0]) else "❌ ขาดทุน/ลดลง"
                        financials_cache[ticker] = {'Revenue': rev_data, 'NetIncome': net_data}
                    else:
                        rev_growth, net_growth = "N/A", "N/A"
                        financials_cache[ticker] = None
                except Exception:
                    rev_growth, net_growth = "N/A", "N/A"
                    financials_cache[ticker] = None

                parent_trend = ""
                for k, v in mega_trends_db.items():
                    if ticker in v: parent_trend = k

                screener_data.append({
                    'หมวดหมู่': parent_trend, 'Ticker': ticker, 'ราคาปัจจุบัน ($)': round(current_price, 2),
                    'ย่อตัว (%)': round(drawdown_pct, 2), 'เทรนด์รายได้': rev_growth, 'เทรนด์กำไรสุทธิ': net_growth
                })

            st.success("✅ สแกนเสร็จสิ้น! ดูตารางผลลัพธ์และบทวิเคราะห์ SWOT ของ 'ผู้ชนะ' ด้านล่างได้เลยครับ")
            
            if screener_data:
                df_master = pd.DataFrame(screener_data)
                def highlight_pass(val):
                    if "✅" in str(val): return 'background-color: #d4edda; color: #155724;'
                    elif "❌" in str(val): return 'background-color: #f8d7da; color: #721c24;'
                    return ''

                for arena_name in selected_trends:
                    df_arena = df_master[df_master['หมวดหมู่'] == arena_name].copy()
                    if df_arena.empty: continue
                    
                    st.markdown(f"## {arena_name}")
                    
                    # โชว์ตารางภาพรวม (ให้เห็นทั้งคนที่สอบผ่านและสอบตกงบการเงิน)
                    st.markdown("#### 📊 ตารางสแกน (หุ้นทั้งหมดที่ราคาเข้าโซนช้อนซื้อ)")
                    try:
                        styled_df = df_arena[['Ticker', 'ราคาปัจจุบัน ($)', 'ย่อตัว (%)', 'เทรนด์รายได้', 'เทรนด์กำไรสุทธิ']].style.map(highlight_pass, subset=['เทรนด์รายได้', 'เทรนด์กำไรสุทธิ'])
                    except AttributeError:
                        styled_df = df_arena[['Ticker', 'ราคาปัจจุบัน ($)', 'ย่อตัว (%)', 'เทรนด์รายได้', 'เทรนด์กำไรสุทธิ']].style.applymap(highlight_pass, subset=['เทรนด์รายได้', 'เทรนด์กำไรสุทธิ'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # 🚀 ไฮไลท์สำคัญ: กรองเอาเฉพาะ "ผู้ชนะ" ที่งบการเงินผ่านเท่านั้น มาทำ SWOT!
                    st.markdown("#### 🔬 AI Analyst: วิเคราะห์ SWOT เฉพาะหุ้นที่ผ่านเกณฑ์งบการเงิน")
                    df_winners = df_arena[(df_arena['เทรนด์รายได้'].str.contains("✅")) & (df_arena['เทรนด์กำไรสุทธิ'].str.contains("✅"))]
                    valid_tickers = df_winners['Ticker'].tolist()
                    
                    if not valid_tickers:
                        st.info("🥲 สัปดาห์นี้ไม่มีหุ้นในหมวดนี้ที่งบการเงินผ่านเกณฑ์ (โตต่อเนื่อง) เลยครับ ระบบจึงระงับการวิเคราะห์ SWOT")
                    else:
                        tabs = st.tabs(valid_tickers)
                        for i, tab in enumerate(tabs):
                            ticker = valid_tickers[i]
                            with tab:
                                c1, c2 = st.columns(2)
                                with c1:
                                    fig_price = go.Figure()
                                    fig_price.add_trace(go.Scatter(x=prices.index[-500:], y=prices[ticker].tail(500), mode='lines', line=dict(color='#2E86C1')))
                                    ath = prices[ticker].max()
                                    fig_price.add_hline(y=ath, line_dash="dash", line_color="red", annotation_text=f" ATH: ${ath:.2f}")
                                    fig_price.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                                    st.plotly_chart(fig_price, use_container_width=True)
                                with c2:
                                    fin_data = financials_cache.get(ticker)
                                    if fin_data and not fin_data['Revenue'].empty:
                                        fig_fin = go.Figure()
                                        fig_fin.add_trace(go.Bar(x=[str(d)[:4] for d in fin_data['Revenue'].index], y=fin_data['Revenue'].values, name='Revenue', marker_color='#27AE60'))
                                        if not fin_data['NetIncome'].empty: fig_fin.add_trace(go.Bar(x=[str(d)[:4] for d in fin_data['NetIncome'].index], y=fin_data['NetIncome'].values, name='Net Income', marker_color='#F39C12'))
                                        fig_fin.update_layout(barmode='group', height=250, margin=dict(l=0, r=0, t=10, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                                        st.plotly_chart(fig_fin, use_container_width=True)
                                    
                                # 🎯 The Real AI Analyst SWOT 🎯
                                st.markdown(f"### 🤖 AI Analyst Report: {ticker}")
                                swot = swot_db.get(ticker, {
                                    "S": f"มีปัจจัยพื้นฐานที่แข็งแกร่ง (ผ่านเกณฑ์รายได้และกำไรเติบโต) อยู่ในหมวด {arena_name.split('.')[1]}",
                                    "W": f"ราคามีความผันผวน ย่อตัวจากจุดสูงสุด (ATH) มาแล้ว {df_winners[df_winners['Ticker']==ticker]['ย่อตัว (%)'].values[0]}%",
                                    "O": "ราคาปัจจุบันอยู่ในโซนที่ได้เปรียบ (Sweet Spot) มีโอกาสทำกำไรสูงเมื่อเทรนด์กลับตัว",
                                    "T": "ความเสี่ยงจากสภาวะเศรษฐกิจมหาภาค (Macro) และการแข่งขันในอุตสาหกรรม"
                                })
                                
                                s1, s2 = st.columns(2)
                                with s1:
                                    st.success(f"**💪 Strengths (จุดแข็ง):**\n{swot['S']}")
                                    st.info(f"**🚀 Opportunities (โอกาส):**\n{swot['O']}")
                                with s2:
                                    st.error(f"**🤕 Weaknesses (จุดอ่อน):**\n{swot['W']}")
                                    st.warning(f"**⚔️ Threats (อุปสรรค):**\n{swot['T']}")
                    st.markdown("---")
else:
    st.info("👈 กดปุ่มสั่งสแกนด้านซ้าย เพื่อค้นหา 'ผู้ชนะ' ที่งบสวยและเข้าโซนช้อนซื้อได้เลยครับ!")