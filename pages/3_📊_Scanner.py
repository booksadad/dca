import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 🛠️ ฟังก์ชันคณิตศาสตร์ (Factor Scoring)
# ==========================================
def calc_zscore(series):
    if series.std() == 0: return series - series.mean()
    return (series - series.mean()) / series.std()

@st.cache_data(ttl=3600) # Cache ข้อมูล 1 ชั่วโมง ป้องกัน API โดนแบนและโหลดเร็วขึ้น
def get_macro_data():
    macro_tickers = {"^VIX": "VIX (ความกลัว)", "^TNX": "US10Y (ดอกเบี้ยพันธบัตร)", "DX-Y.NYB": "DXY (ดัชนีดอลลาร์)"}
    data = yf.download(list(macro_tickers.keys()), period="1mo", progress=False)['Close']
    
    results = {}
    for t, name in macro_tickers.items():
        if t in data:
            current = data[t].iloc[-1]
            prev = data[t].iloc[-2]
            pct_change = ((current - prev) / prev) * 100
            results[name] = {"val": current, "change": pct_change}
    return results

@st.cache_data(ttl=3600)
def scan_alpha_universe(universe):
    # 1. ดึงราคาย้อนหลังเพื่อหา Momentum & Volatility
    prices = yf.download(universe, period="1y", progress=False)['Close']
    returns = prices.pct_change().dropna()
    
    ret_3m = prices.pct_change(periods=63).iloc[-1]
    ret_6m = prices.pct_change(periods=126).iloc[-1]
    vol = returns.tail(126).std() * np.sqrt(252)
    
    # Risk-Adjusted Momentum
    mom_score = ((ret_3m * 0.4) + (ret_6m * 0.6)) / vol 
    
    df = pd.DataFrame({'Ticker': prices.columns, 'Momentum': mom_score, 'Volatility': vol}).dropna()
    
    # 2. ดึงงบการเงิน (Fundamentals)
    metrics = []
    for t in df['Ticker']:
        try:
            info = yf.Ticker(t).info
            roa = info.get('returnOnAssets', np.nan)
            margin = info.get('profitMargins', np.nan)
            peg = info.get('pegRatio', np.nan)
            # FCF Yield = Free Cash Flow / Market Cap
            fcf = info.get('freeCashflow', np.nan)
            mcap = info.get('marketCap', np.nan)
            fcf_yield = (fcf / mcap) if pd.notna(fcf) and pd.notna(mcap) and mcap > 0 else np.nan
            
            metrics.append({
                'Ticker': t, 
                'ROA': roa * 100 if pd.notna(roa) else np.nan,
                'Margin': margin * 100 if pd.notna(margin) else np.nan,
                'PEG': peg,
                'FCF_Yield': fcf_yield * 100 if pd.notna(fcf_yield) else np.nan
            })
        except:
            metrics.append({'Ticker': t, 'ROA': np.nan, 'Margin': np.nan, 'PEG': np.nan, 'FCF_Yield': np.nan})
            
    df_metrics = pd.DataFrame(metrics)
    final_df = pd.merge(df, df_metrics, on='Ticker')
    
    # เติมค่าว่างด้วยค่ามัธยฐานของกลุ่มเพื่อไม่ให้สมการพัง
    for col in ['ROA', 'Margin', 'PEG', 'FCF_Yield']:
        final_df[col] = final_df[col].fillna(final_df[col].median())
        
    # 3. ให้คะแนน Alpha Score (The 0.1% Formula)
    # PEG ยิ่งต่ำยิ่งดี เลยต้องคูณ -1 กลับทิศทาง Z-Score
    final_df['Z_Mom'] = calc_zscore(final_df['Momentum'])
    final_df['Z_Quality'] = (calc_zscore(final_df['ROA']) + calc_zscore(final_df['Margin'])) / 2
    final_df['Z_Value'] = (calc_zscore(final_df['FCF_Yield']) + (calc_zscore(final_df['PEG']) * -1)) / 2
    
    # ถ่วงน้ำหนักคะแนน (บุก 50%, คุณภาพ 30%, ความถูกแพง 20%)
    final_df['Total_Alpha_Score'] = (final_df['Z_Mom'] * 0.5) + (final_df['Z_Quality'] * 0.3) + (final_df['Z_Value'] * 0.2)
    
    return final_df.sort_values('Total_Alpha_Score', ascending=False).reset_index(drop=True)

# ==========================================
# 🖥️ หน้าจอหลัก (UI)
# ==========================================
st.set_page_config(page_title="Alpha Scanner", page_icon="🔭", layout="wide")
st.title("🔭 QUANT-HQ: The Alpha Scanner")
st.markdown("เรดาร์สแกนหา **จ่าฝูง (Alpha)** ด้วยสมการ Multi-Factor (Momentum + Quality + Value)")
st.markdown("---")

# 📡 1. สภาพอากาศระดับมหภาค (Macro Weather Station)
st.markdown("### 📡 1. สภาพอากาศระดับมหภาค (Macro Weather)")
macro_data = get_macro_data()
cols = st.columns(3)
for i, (name, data) in enumerate(macro_data.items()):
    with cols[i]:
        st.metric(label=name, value=f"{data['val']:.2f}", delta=f"{data['change']:.2f}%", delta_color="inverse" if "VIX" in name or "US10Y" in name else "normal")

st.markdown("---")

# 🧬 2. กำหนดขอบเขตเรดาร์ (Universe Selection)
st.markdown("### 🧬 2. สแกนตะแกรงร่อน Factor Investing")
st.markdown("ระบบจะดึงข้อมูลงบการเงินและกราฟย้อนหลังมาคำนวณ Z-Score กรุณาเลือกกลุ่มหุ้นที่ต้องการสแกน:")

# หุ้นกลุ่มซิ่งและน่าสนใจในตลาดสหรัฐฯ (ปรับแต่งได้)
UNIVERSE_PRESETS = {
    "🔥 Aggressive Tech & AI (สายบุก)": ["NVDA", "AMD", "AVGO", "SMCI", "PLTR", "CRWD", "ARM", "TSM", "ASML", "MSFT", "META", "GOOGL"],
    "🚀 Disruptive Growth (อนาคต)": ["RKLB", "TSLA", "MSTR", "COIN", "MELI", "SHOP", "SPOT", "UBER"],
    "🛡️ Quality Compounders (ตัวแบกพอร์ต)": ["COST", "V", "MA", "LLY", "UNH", "JPM", "BRK-B", "BLK", "AAPL", "AMZN"]
}

selected_preset = st.radio("เป้าหมายเรดาร์:", list(UNIVERSE_PRESETS.keys()), horizontal=True)
custom_tickers = st.text_input("หรือพิมพ์ชื่อหุ้นที่ต้องการสแกนเอง (คั่นด้วยลูกน้ำ):", "")

if st.button("🚀 เดินเครื่องเรดาร์สแกนตลาด", type="primary"):
    with st.spinner("⏳ กำลังกวาดข้อมูลจาก Wall Street และรันสมการ Z-Score... (อาจใช้เวลา 10-20 วินาที)"):
        
        scan_list = [t.strip().upper() for t in custom_tickers.split(",")] if custom_tickers else UNIVERSE_PRESETS[selected_preset]
        scan_list = list(set([t for t in scan_list if t])) # ลบตัวซ้ำ
        
        df_alpha = scan_alpha_universe(scan_list)
        
        st.success(f"✅ สแกนเสร็จสิ้น! พบหุ้นผ่านเกณฑ์ {len(df_alpha)} ตัว")
        
        # จัดรูปแบบตารางให้ดูง่าย
        display_df = df_alpha[['Ticker', 'Total_Alpha_Score', 'Momentum', 'ROA', 'Margin', 'PEG', 'FCF_Yield']].copy()
        display_df.columns = ['หุ้น', 'Alpha Score', 'Momentum (ความซิ่ง)', 'ROA (%)', 'Margin (%)', 'PEG Ratio', 'FCF Yield (%)']
        for col in display_df.columns[1:]:
            display_df[col] = display_df[col].round(2)
            
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        
        # ไฮไลต์ Top 3
        st.dataframe(display_df.style.highlight_max(subset=['Alpha Score'], color='#1f77b4'), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🗺️ แผนที่พิกัด (Risk vs Reward)")
        
        # วาดกราฟ Scatter Plot ให้เห็นภาพรวม
        fig = px.scatter(
            df_alpha, x="Z_Quality", y="Z_Mom", text="Ticker", size="Total_Alpha_Score", 
            color="Total_Alpha_Score", color_continuous_scale="Plasma",
            labels={"Z_Quality": "คุณภาพกิจการ (Quality Z-Score)", "Z_Mom": "ความซิ่งทะลุจอ (Momentum Z-Score)"},
            title="เรดาร์แยกแยะ: หุ้นไหนของแท้ หุ้นไหนปั่น (ขวาบน = สุดยอด)"
        )
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        # ขีดเส้นแกน 0 แบ่งโซน
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
