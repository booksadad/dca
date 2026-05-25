import streamlit as st

st.set_page_config(page_title="AI Copilot", page_icon="🤖")
st.title("🤖 QuantHQ : AI Copilot")
st.markdown("---")

# 🛡️ ด่านที่ 1: เช็กว่าติดตั้งไลบรารี google-generativeai ใน requirements.txt หรือยัง?
try:
    import google.generativeai as genai
except ImportError:
    st.error("🚨 ระบบหาเครื่องมือ AI ไม่เจอ! (ImportError)")
    st.warning("👉 โปรดไปที่ GitHub เปิดไฟล์ `requirements.txt` แล้วพิมพ์คำว่า `google-generativeai` เพิ่มลงไปบรรทัดล่างสุดครับ")
    st.stop() # หยุดการทำงานชั่วคราวเพื่อไม่ให้แอปพัง

# 🛡️ ด่านที่ 2: เช็กว่าตู้เซฟ Secrets มีกุญแจอยู่จริงไหม?
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("❌ หา API Key ในตู้เซฟไม่เจอ! (Secrets Error)")
    st.warning("👉 โปรดไปที่หน้าเว็บ Streamlit -> Settings -> Secrets แล้วตั้งค่า `GEMINI_API_KEY = \"กุญแจของคุณ\"`")
    st.stop()

# 🛡️ ด่านที่ 3: ถ้าผ่าน 2 ด่านแรกมาได้ ให้รันระบบ AI
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    st.success("✅ เชื่อมต่อสมองกล AI สำเร็จ พร้อมใช้งาน!")
    
    ticker = st.text_input("🔍 พิมพ์ชื่อหุ้นที่ต้องการให้ AI วิเคราะห์ (เช่น MSFT, RKLB, V)", "MSFT")
    
    if st.button("🧠 สแกนเจาะลึก!", type="primary"):
        with st.spinner(f"กำลังดึงข้อมูลลับของ {ticker.upper()}..."):
            # Prompt สำหรับวิเคราะห์หุ้น
            prompt = f"""
            ในฐานะนักวิเคราะห์ Quant ระดับสถาบัน 
            ช่วยสรุปข้อมูลหุ้น {ticker.upper()} แบบสั้น กระชับ เป็น Bullet Points ตามนี้:
            1. จุดแข็งของโมเดลธุรกิจ (Earning Power)
            2. ความเสี่ยงที่ต้องระวัง (Max Drawdown triggers)
            """
            response = model.generate_content(prompt)
            st.info(response.text)
            
except Exception as e:
    # ด่านสุดท้าย: ถ้าคีย์ผิด หรือเน็ตหลุด จะฟ้องตรงนี้
    st.error(f"❌ AI เกิดอาการช็อตกะทันหัน: {e}")
