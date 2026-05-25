import streamlit as st

st.set_page_config(page_title="AI Copilot", page_icon="🤖")
st.title("🤖 QuantHQ : AI Copilot")
st.markdown("---")

# 🛡️ ด่านที่ 1: เช็กไลบรารี
try:
    import google.generativeai as genai
except ImportError:
    st.error("🚨 ระบบหาเครื่องมือ AI ไม่เจอ! โปรดเช็ก requirements.txt")
    st.stop()

# 🛡️ ด่านที่ 2: เช็ก API Key
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("❌ หา API Key ในตู้เซฟไม่เจอ! โปรดเช็ก Streamlit Secrets")
    st.stop()

# 🛡️ ด่านที่ 3: รันระบบสมองกล
try:
    genai.configure(api_key=api_key)
    # ใช้เครื่องยนต์ 2.0 ที่แรงและเสถียรที่สุดตอนนี้
    model = genai.GenerativeModel('gemini-3.1-flash-lite') 
    
    st.success("✅ เชื่อมต่อสมองกล AI สำเร็จ พร้อมใช้งาน!")
    
    ticker = st.text_input("🔍 พิมพ์ชื่อหุ้นที่ต้องการให้ AI วิเคราะห์ (เช่น MSFT, RKLB, V)", "MSFT")
    
    if st.button("🧠 สแกนเจาะลึก!", type="primary"):
        with st.spinner(f"กำลังสกัดข้อมูลแก่นแท้ของ {ticker.upper()}..."):
            # วิศวกรรมคำสั่ง (Prompt) ฉบับ Quant โหดๆ กระชับๆ
            prompt = f"""
            ในฐานะ Chief Investment Officer (CIO) สาย Quant 
            โปรดวิเคราะห์หุ้น {ticker.upper()} สำหรับพอร์ต DCA (งบลงทุนรอบละ 500 บาท, ขั้นต่ำการยิงออเดอร์ 50 บาท/หุ้น)
            ขอเนื้อหาที่อัดแน่นด้วย Insight เชิงลึก มีตัวเลขสถิติประกอบ และจัดหน้าแบบ Bullet ให้อ่านง่าย (ห้ามเกริ่นนำ)

            ตอบเรียงตาม 3 หัวข้อนี้:
            1. 💰 แหล่งผลิตกระแสเงินสด & คู่แข่ง (Earning Power & Moat): 
               - สรุปโมเดลธุรกิจ และระบุตัวเลขทางการเงินที่สำคัญ (เช่น Market Cap, Margin หรือ P/E ถ้ารู้)
               - ใครคือคู่แข่งเบอร์ 1 และความได้เปรียบของเราคืออะไร?
            2. 📉 หลุมยุบความเสี่ยง (Max Drawdown Triggers): 
               - ปัจจัยที่จะทำให้หุ้นร่วงแรงที่สุดคืออะไร? 
               - หุ้นตัวนี้มีความผันผวนสูงแค่ไหน? (ระบุค่า Beta หรือ Cash Burn rate โดยประมาณ)
            3. 🎯 แผนจัดสรรเงิน 500 บาท (Actionable Allocation): 
               - ควรจัดหุ้นนี้เป็นกลุ่มไหน? (Alpha ซิ่ง / Compounder ทบต้น / Defensive กันชน)
               - ด้วยข้อจำกัดงบ 500 บาท และขั้นต่ำ 50 บาท... ควรแบ่งเงินมาซื้อตัวนี้กี่บาทในรอบนี้? หรือควรใช้กลยุทธ์อื่น (เช่น สะสมเงินไว้ซื้อเดือนละครั้ง) หรือควรข้ามไปก่อน? ให้ฟันธงมาเลย!
            """
            
            response = model.generate_content(prompt)
            st.info(response.text)
            
except Exception as e:
    st.error(f"❌ AI เกิดอาการช็อตกะทันหัน: {e}")
