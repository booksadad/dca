import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="AI Copilot", page_icon="🤖")

st.title("🤖 QuantHQ : AI Copilot")
st.markdown("ผู้ช่วยวิเคราะห์ข่าวและงบการเงิน")
st.markdown("---")

# 🧠 อัปเกรด: ใช้ .get เพื่อป้องกันโค้ดเอ๋อจนหน้าเมนูหายไปเฉยๆ
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("❌ ไม่พบคีย์ในตู้เซฟหลังบ้าน! โปรดไปที่ Settings -> Secrets บนหน้าเว็บ Streamlit Cloud แล้วพิมพ์ใส่ลงไป: GEMINI_API_KEY = 'คีย์ของกัปตัน'")
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        ticker = st.text_input("พิมพ์ชื่อหุ้นที่ต้องการให้ AI วิเคราะห์", "MSFT")
        if st.button("🧠 วิเคราะห์เลย!", type="primary"):
            with st.spinner("AI กำลังวิเคราะห์ข้อมูล..."):
                prompt = f"สรุปจุดแข็งและความเสี่ยงของหุ้น {ticker.upper()} มาสั้นๆ กระชับ 3 Bullet points"
                response = model.generate_content(prompt)
                st.info(response.text)
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ AI: {e}")
