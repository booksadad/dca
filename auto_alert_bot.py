import os
import requests
import yfinance as yf
from datetime import datetime
import pytz

# ดึงกุญแจความลับจาก GitHub Secrets
DISCORD_URL = os.environ.get("DISCORD_WEBHOOK_URL")

def get_market_pulse():
    try:
        vix = yf.download("^VIX", period="1d", progress=False)['Close'].iloc[-1]
        return float(vix)
    except:
        return 20.0

def get_fx_rate():
    try:
        fx = yf.download("THB=X", period="1d", progress=False)['Close'].iloc[-1]
        return float(fx)
    except:
        return 36.50

if __name__ == "__main__":
    if not DISCORD_URL:
        print("🚨 ไม่พบ DISCORD_WEBHOOK_URL ในระบบ หยุดการทำงาน")
        exit()

    vix_val = get_market_pulse()
    fx_val = get_fx_rate()
    # ดึงเวลาประเทศไทย (Bangkok)
    bkk_time = datetime.now(pytz.timezone('Asia/Bangkok')).strftime('%Y-%m-%d %H:%M')
    
    market_mood = "🔴 ตลาดแพนิคผันผวนสูง (ควรระวัง)" if vix_val > 25 else "🟢 ตลาดปกติ (ทยอยสะสมได้)"
    color_code = 15158332 if vix_val > 25 else 3066993

    embed_data = {
        "username": "QuantHQ Scheduler",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/2103/2103832.png",
        "content": "🔔 **ตื่นได้แล้วกัปตัน! ถึงรอบประจำการ DCA 500 บาทแล้ว (พุธ/เสาร์)**",
        "embeds": [
            {
                "title": "🛡️ QuantHQ - Market Radar (Morning Brief)",
                "color": color_code,
                "fields": [
                    {"name": "🌡️ ชีพจรตลาด (VIX)", "value": f"{vix_val:.2f}\n{market_mood}", "inline": True},
                    {"name": "💱 อัตราแลกเปลี่ยน", "value": f"{fx_val:.2f} THB/USD", "inline": True},
                    {"name": "🎯 แผนปฏิบัติการเช้านี้", "value": "1. เปิดเว็บ Streamlit ของเรา\n2. กดรัน Quant Matrix\n3. กดยืนยันออเดอร์ผ่านแอปโบรคเกอร์", "inline": False}
                ],
                "footer": {"text": f"Scanned automatically at: {bkk_time} (BKK)"}
            }
        ]
    }

    resp = requests.post(DISCORD_URL, json=embed_data)
    if resp.status_code == 204:
        print("✅ ส่งแจ้งเตือนภารกิจเช้าสำเร็จ!")
    else:
        print(f"❌ ส่งไม่สำเร็จ: {resp.status_code}")
