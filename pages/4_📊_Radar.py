import streamlit as st
import yfinance as yf
from datetime import datetime
from deep_translator import GoogleTranslator

# ==========================================
# 🖥️ หน้าจอหลักของโมดูล News Radar
# ==========================================
st.set_page_config(page_title="QuantHQ | News Radar", page_icon="📰", layout="wide")

st.title("📰 Stock News & Earnings Radar")
st.markdown("ระบบสแกนความเคลื่อนไหว เจาะลึกปัจจัยพื้นฐาน พร้อม AI แปลและสรุปข่าวเรียลไทม์")
st.markdown("---")

# แถบตั้งค่าด้านซ้ายมือ (เพื่อให้ดึงรายชื่อหุ้นมาใช้ได้เลย)
st.sidebar.subheader("🗂️ ตั้งค่าจักรวาลการเทรด")
tickers_input = st.sidebar.text_area("รายชื่อหุ้นในพอร์ต (คั่นด้วยลูกน้ำ)", "NVDA, GOOG, KO, TSLA, SPG, MO, JNJ, AAPL, BLK")
my_portfolio = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.sidebar.markdown("---")

if not my_portfolio:
    st.warning("⚠️ โปรดระบุชื่อหุ้นในช่องกรอกข้อมูลด้านซ้ายมือก่อนครับ")
else:
    st.info("💡 เคล็ดลับ: แวะเข้ามาเช็คหน้านี้เพื่อดูข่าวสารและตัวเลขงบการเงินล่าสุด ก่อนตัดสินใจอัดฉีดงบเข้าพอร์ตหน้า DCA")
    
    # สร้างแฟ้มข้อมูล (Tabs) แยกตามรายชื่อหุ้น
    tabs = st.tabs(my_portfolio)
    
    # 🤖 เรียกใช้งาน AI วุ้นแปลภาษา (ตั้งค่าครั้งเดียวใช้ได้ทั้งหน้า)
    translator = GoogleTranslator(source='auto', target='th')
    
    for i, ticker in enumerate(my_portfolio):
        with tabs[i]:
            st.subheader(f"📊 สรุปข้อมูลของบริษัท {ticker}")
            
            # โหลดข้อมูลด้วย Spinner เพื่อไม่ให้เว็บค้าง
            with st.spinner(f"กำลังดึงข้อมูลลับจาก Wall Street สำหรับ {ticker}..."):
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # 💰 ส่วนที่ 1: งบการเงิน (Financial Metrics)
                    st.markdown("**💰 อัตราส่วนทางการเงิน (Financial Ratios)**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    pe_ratio = info.get('trailingPE')
                    fwd_pe = info.get('forwardPE')
                    margin = info.get('profitMargins')
                    roe = info.get('returnOnEquity')
                    
                    col1.metric("P/E Ratio (ปัจจุบัน)", f"{round(pe_ratio, 2)} เท่า" if pe_ratio else "N/A")
                    col2.metric("Forward P/E (คาดการณ์)", f"{round(fwd_pe, 2)} เท่า" if fwd_pe else "N/A")
                    col3.metric("Profit Margin (กำไรสุทธิ)", f"{round(margin * 100, 2)}%" if margin else "N/A")
                    col4.metric("ROE (ผลตอบแทนผู้ถือหุ้น)", f"{round(roe * 100, 2)}%" if roe else "N/A")
                    
                    st.markdown("---")
                    
                    # 📰 ส่วนที่ 2: ข่าวพาดหัวล่าสุด (พร้อม AI แปลไทย)
                    st.markdown("**📰 พาดหัวข่าวล่าสุด (แปลไทย & สรุปย่อ)**")
                    news_list = stock.news
                    
                    if news_list and len(news_list) > 0:
                        valid_news_count = 0
                        
                        for n in news_list:
                            if valid_news_count >= 5: break # ดึงมาโชว์แค่ 5 ข่าวล่าสุดเพื่อความรวดเร็ว
                            
                            try:
                                # 🛡️ ระบบถอดรหัสข่าวแบบกันพัง (รองรับ API ทุกรูปแบบ)
                                if 'content' in n:
                                    c = n['content']
                                    title = c.get('title', 'No Title')
                                    summary = c.get('summary', '')
                                    
                                    link_data = c.get('clickThroughUrl', c.get('canonicalUrl', {}))
                                    if isinstance(link_data, dict):
                                        link = link_data.get('url', '#')
                                    else:
                                        link = str(link_data) if link_data else '#'
                                        
                                    publisher = c.get('provider', {}).get('displayName', 'Unknown Source')
                                    pub_date = c.get('pubDate', '')
                                    date_str = str(pub_date)[:10] if pub_date else "Recent"
                                    
                                else:
                                    title = n.get('title', 'No Title')
                                    summary = n.get('summary', '')
                                    
                                    link = n.get('link', '#')
                                    if isinstance(link, dict): link = link.get('url', '#')
                                    
                                    publisher = n.get('publisher', 'Unknown Source')
                                    pub_time = n.get('providerPublishTime')
                                    date_str = datetime.fromtimestamp(int(pub_time)).strftime('%d %b %Y | %H:%M') if pub_time else "Recent"
                                
                                if title == 'No Title': continue 
                                
                                # 🌐 ให้ AI ทำการแปลภาษา
                                try:
                                    title_th = translator.translate(title)
                                    summary_th = translator.translate(summary) if summary else ""
                                except Exception:
                                    # ถ้าแปลพัง ให้โชว์ภาษาอังกฤษเดิมไปเลย เว็บจะได้ไม่ค้าง
                                    title_th = title
                                    summary_th = summary
                                
                                # 🖥️ แสดงผลข่าว
                                st.markdown(f"👉 **[{title_th}]({link})**")
                                st.caption(f"✍️ สำนักข่าว: {publisher} | 🕒 เผยแพร่เมื่อ: {date_str}")
                                
                                if summary_th:
                                    st.info(f"💡 **สรุปย่อ:** {summary_th}")
                                else:
                                    st.markdown(" ") 
                                    
                                valid_news_count += 1
                                
                            except Exception as e:
                                continue # ข้ามข่าวที่โครงสร้างพัง
                            
                        if valid_news_count == 0:
                            st.write("ระบบฐานข้อมูลข่าวของ Wall Street กำลังปรับปรุงชั่วคราวครับ")
                    else:
                        st.write("ไม่มีความเคลื่อนไหวทางข่าวสารที่สำคัญในช่วงนี้ครับ")
                        
                except Exception as e:
                    st.error(f"❌ ไม่สามารถโหลดข้อมูลของ {ticker} ได้ชั่วคราว (YFinance API อาจจะติดข้อจำกัด)")
