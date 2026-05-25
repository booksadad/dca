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
            โปรดวิเคราะห์เจาะลึกหุ้น {ticker.upper()} เพื่อประกอบการตัดสินใจ DCA 
            ขอเนื้อหาที่อัดแน่นด้วย Insight เชิงลึก มีการอ้างอิงตัวเลขหรือคู่แข่ง แต่จัดหน้าให้อ่านง่าย (ห้ามเกริ่นนำ)

            ตอบเรียงตาม 3 หัวข้อนี้:
            1. 💰 แหล่งผลิตกระแสเงินสด (Earning Power & Moat): 
               - อธิบายโมเดลธุรกิจหลักที่ทำกำไร
               - อำนาจการผูกขาด (Moat) หรือความได้เปรียบเหนือคู่แข่งคืออะไร? (ยกตัวอย่างประกอบ)
            2. 📉 จุดตายและหลุมยุบ (Drawdown Triggers): 
               - ปัจจัยเศรษฐกิจ นโยบาย หรือเทคโนโลยีอะไร ที่จะทำให้หุ้นตัวนี้ราคาตกต่ำอย่างรุนแรง (Worst-case scenario)?
            3. 🎯 บทบาทในพอร์ต DCA (Portfolio Allocation): 
               - จัดเป็นกลุ่มไหน? (Alpha ซิ่งทำกำไร / Compounder ทบต้น / Defensive กันชน)
               - เหมาะกับสัดส่วนน้ำหนักในพอร์ตมากหรือน้อย เพราะเหตุใด?
            """
            
            response = model.generate_content(prompt)
            st.info(response.text)
            
except Exception as e:
    st.error(f"❌ AI เกิดอาการช็อตกะทันหัน: {e}")
