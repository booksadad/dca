"""
🔔 discord_notifier.py — QuantHQ Discord Notification System
=============================================================
ระบบแจ้งเตือน Discord แบบครบวงจรผ่าน Webhook

Features:
- 📊 สรุปคำสั่ง DCA ทุกรอบ (Execution Summary)
- 🚨 แจ้งเตือน Risk (Drawdown, Concentration, VIX Spike)
- 🔄 แจ้งเตือน Regime Change (Bull → Bear)
- 📈 สรุปพอร์ตรายวัน (Daily Snapshot)
- ⚠️ แจ้งเตือน Factor Health (IC/ICIR)
- 🛑 Circuit Breaker Alert

วิธีใช้:
    1. สร้าง Webhook ใน Discord Server → Server Settings → Integrations → Webhooks
    2. ใส่ URL ไว้ใน st.secrets["DISCORD_WEBHOOK_URL"] หรือ environment variable
    3. Import แล้วเรียกใช้:
       from discord_notifier import DiscordNotifier
       notifier = DiscordNotifier(webhook_url)
       notifier.send_dca_summary(...)
"""

import requests
import json
import os
from datetime import datetime, timezone


class DiscordNotifier:
    """ตัวจัดการแจ้งเตือน Discord ผ่าน Webhook"""

    def __init__(self, webhook_url=None):
        """
        Args:
            webhook_url: Discord Webhook URL
                         ถ้าไม่ใส่จะดึงจาก st.secrets หรือ env var
        """
        self.webhook_url = webhook_url or self._get_webhook_url()
        self.enabled = bool(self.webhook_url)

    @staticmethod
    def _get_webhook_url():
        """ดึง Webhook URL จาก Environment variable"""
        return os.environ.get("DISCORD_WEBHOOK_URL", "")

    def _send(self, payload):
        """ส่ง payload ไปยัง Discord Webhook"""
        if not self.enabled:
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if response.status_code == 204:
                return True
            else:
                print(f"Discord webhook error: {response.status_code} — {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Discord connection error: {e}")
            return False

    # ==========================================
    # 📊 1. สรุปคำสั่ง DCA (Execution Summary)
    # ==========================================
    def send_dca_summary(self, execution_df, regime_info, budget, fx_rate=None, is_dry_run=False):
        """
        ส่งสรุปคำสั่ง DCA ทั้งรอบไป Discord

        Args:
            execution_df: DataFrame จากตาราง Quant Allocation
                          ต้องมี columns: หุ้น, เหตุผล, เป้า%, ซื้อ, ขาย
            regime_info: dict เช่น {"state": "BULL", "p_bull": 0.95, "p_panic": 0.05}
            budget: งบประมาณรอบนี้ (บาท)
            fx_rate: อัตราแลกเปลี่ยน USD/THB (optional)
            is_dry_run: (bool) ถ้า True จะแสดงว่าเป็น Forward Test
        """
        now = datetime.now().strftime("%d/%m/%Y %H:%M")

        # แยก Buy / Sell / Hold
        buys = execution_df[execution_df["ซื้อ"] > 0] if "ซื้อ" in execution_df.columns else []
        sells = execution_df[execution_df["ขาย"] > 0] if "ขาย" in execution_df.columns else []

        total_buy = execution_df["ซื้อ"].sum() if "ซื้อ" in execution_df.columns else 0
        total_sell = execution_df["ขาย"].sum() if "ขาย" in execution_df.columns else 0

        # สร้าง Buy Lines
        buy_lines = ""
        if len(buys) > 0:
            for _, row in buys.iterrows():
                ticker = row.get("หุ้น", row.get("Ticker", "?"))
                amount = row["ซื้อ"]
                target = row.get("เป้า%", row.get("Target_%", "?"))
                buy_lines += f"🟢 **{ticker}** — ซื้อ ฿{amount:,.0f} (เป้า {target}%)\n"
        else:
            buy_lines = "— ไม่มีรายการซื้อรอบนี้ —\n"

        # สร้าง Sell Lines
        sell_lines = ""
        if len(sells) > 0:
            for _, row in sells.iterrows():
                ticker = row.get("หุ้น", row.get("Ticker", "?"))
                amount = row["ขาย"]
                reason = row.get("เหตุผล", row.get("Action", ""))
                sell_lines += f"🔴 **{ticker}** — ขาย ฿{amount:,.0f} ({reason})\n"
        else:
            sell_lines = "— ไม่มีรายการขายรอบนี้ —\n"

        # Regime emoji
        regime_state = regime_info.get("state", "UNKNOWN")
        regime_emoji = {"BULL": "🐂", "BEAR": "🐻", "PANIC": "🚨"}.get(regime_state, "❓")
        p_bull = regime_info.get("p_bull", 0)
        p_panic = regime_info.get("p_panic", 0)

        title_prefix = "[FORWARD TEST] " if is_dry_run else ""
        
        embed = {
            "title": f"{title_prefix}📊 QuantHQ — สรุปคำสั่ง DCA",
            "description": f"**วันที่**: {now}\n**งบประมาณ**: ฿{budget:,.0f}",
            "color": 0x95a5a6 if is_dry_run else (0x2ECC71 if total_buy > total_sell else 0xE74C3C),
            "fields": [
                {
                    "name": f"{regime_emoji} Regime: {regime_state}",
                    "value": f"P(Bull): {p_bull:.1%} | P(Panic): {p_panic:.1%}",
                    "inline": False,
                },
                {
                    "name": "🟢 รายการซื้อ",
                    "value": buy_lines[:1024],
                    "inline": True,
                },
                {
                    "name": "🔴 รายการขาย",
                    "value": sell_lines[:1024],
                    "inline": True,
                },
                {
                    "name": "💰 สรุปยอด",
                    "value": f"ซื้อรวม: ฿{total_buy:,.0f}\nขายรวม: ฿{total_sell:,.0f}",
                    "inline": False,
                },
                {
                    "name": "📱 แอคชั่นสำหรับ Dime",
                    "value": "กรุณาเปิดแอป Dime บนมือถือ แล้วส่งคำสั่งตามรายการด้านบนครับ 🎯",
                    "inline": False,
                }
            ],
            "footer": {"text": "QuantHQ Terminal — Institutional DCA Engine"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if fx_rate:
            embed["fields"].append(
                {
                    "name": "💱 FX Rate",
                    "value": f"USD/THB = {fx_rate:.2f}",
                    "inline": True,
                }
            )

        return self._send({"embeds": [embed]})

    # ==========================================
    # 🚨 2. แจ้งเตือน Risk Alert
    # ==========================================
    def send_risk_alert(self, alert_type, details, severity="WARNING"):
        """
        ส่งแจ้งเตือนความเสี่ยงไป Discord

        Args:
            alert_type: ประเภทเตือน เช่น "DRAWDOWN", "VIX_SPIKE", "CONCENTRATION"
            details: รายละเอียด (string)
            severity: "WARNING" / "CRITICAL" / "INFO"
        """
        color_map = {
            "CRITICAL": 0xE74C3C,  # แดง
            "WARNING": 0xF39C12,   # ส้ม
            "INFO": 0x3498DB,      # ฟ้า
        }
        emoji_map = {
            "DRAWDOWN": "📉",
            "VIX_SPIKE": "😨",
            "CONCENTRATION": "⚖️",
            "CIRCUIT_BREAKER": "🛑",
            "DATA_STALE": "📡",
            "FACTOR_DECAY": "📊",
        }

        embed = {
            "title": f"{emoji_map.get(alert_type, '⚠️')} Risk Alert: {alert_type}",
            "description": details,
            "color": color_map.get(severity, 0xF39C12),
            "fields": [
                {"name": "ระดับ", "value": severity, "inline": True},
                {"name": "เวลา", "value": datetime.now().strftime("%d/%m/%Y %H:%M"), "inline": True},
            ],
            "footer": {"text": "QuantHQ Risk Engine"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # CRITICAL → mention @everyone
        content = ""
        if severity == "CRITICAL":
            content = "🚨 **@everyone** — Risk Alert ระดับวิกฤต!"

        return self._send({"content": content, "embeds": [embed]})

    # ==========================================
    # 🔄 3. แจ้งเตือน Regime Change
    # ==========================================
    def send_regime_change(self, old_regime, new_regime, probabilities, factor_weights):
        """
        ส่งแจ้งเตือนเมื่อ HMM Regime เปลี่ยน

        Args:
            old_regime: สถานะเดิม เช่น "BULL"
            new_regime: สถานะใหม่ เช่น "BEAR"
            probabilities: dict เช่น {"P_BULL": 0.3, "P_PANIC": 0.7}
            factor_weights: dict เช่น {"Mom": 0.15, "Qual": 0.50, "Val": 0.35}
        """
        transition_emoji = {
            ("BULL", "BEAR"): "🐂 → 🐻",
            ("BEAR", "BULL"): "🐻 → 🐂",
            ("BULL", "PANIC"): "🐂 → 🚨",
            ("PANIC", "BULL"): "🚨 → 🐂",
            ("BEAR", "PANIC"): "🐻 → 🚨",
            ("PANIC", "BEAR"): "🚨 → 🐻",
        }
        emoji = transition_emoji.get((old_regime, new_regime), "🔄")
        color = 0xE74C3C if new_regime in ["BEAR", "PANIC"] else 0x2ECC71

        fw_text = " | ".join([f"{k}: {v:.0%}" for k, v in factor_weights.items()])

        embed = {
            "title": f"{emoji} Regime Change Detected!",
            "description": f"ตลาดเปลี่ยนจาก **{old_regime}** → **{new_regime}**",
            "color": color,
            "fields": [
                {
                    "name": "📊 Probabilities",
                    "value": f"P(Bull): {probabilities.get('P_BULL', 0):.1%}\nP(Panic): {probabilities.get('P_PANIC', 0):.1%}",
                    "inline": True,
                },
                {
                    "name": "⚡ Factor Weights ใหม่",
                    "value": fw_text,
                    "inline": True,
                },
                {
                    "name": "💡 คำแนะนำ",
                    "value": self._get_regime_advice(new_regime),
                    "inline": False,
                },
            ],
            "footer": {"text": "QuantHQ HMM Regime Engine"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return self._send({"embeds": [embed]})

    @staticmethod
    def _get_regime_advice(regime):
        advice = {
            "BULL": "🟢 ตลาดขาขึ้น — เพิ่มน้ำหนัก Momentum, ลงทุนตามแผน DCA ปกติ",
            "BEAR": "🟡 ตลาดขาลง — เพิ่มน้ำหนัก Quality/Value, ลด position size 20-30%",
            "PANIC": "🔴 ตลาดแพนิค — หยุดซื้อเพิ่ม, ถือ Cash เพิ่ม, ตรวจ Stop-loss",
        }
        return advice.get(regime, "ไม่มีคำแนะนำ")

    # ==========================================
    # 📈 4. สรุปพอร์ตรายวัน (Daily Snapshot)
    # ==========================================
    def send_daily_snapshot(self, portfolio_summary):
        """
        ส่งสรุปพอร์ตรายวัน

        Args:
            portfolio_summary: dict เช่น {
                "total_value": 5000.0,
                "daily_pnl": 120.5,
                "daily_pnl_pct": 2.4,
                "top_gainer": {"ticker": "NVDA", "pct": 3.5},
                "top_loser": {"ticker": "ENPH", "pct": -2.1},
                "regime": "BULL",
                "vix": 15.2,
                "holdings_count": 7,
            }
        """
        ps = portfolio_summary
        pnl = ps.get("daily_pnl", 0)
        pnl_pct = ps.get("daily_pnl_pct", 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        color = 0x2ECC71 if pnl >= 0 else 0xE74C3C

        top_g = ps.get("top_gainer", {})
        top_l = ps.get("top_loser", {})

        embed = {
            "title": f"{pnl_emoji} สรุปพอร์ตประจำวัน",
            "color": color,
            "fields": [
                {
                    "name": "💰 มูลค่าพอร์ต",
                    "value": f"฿{ps.get('total_value', 0):,.0f}",
                    "inline": True,
                },
                {
                    "name": f"{pnl_emoji} P&L วันนี้",
                    "value": f"{'+'if pnl>=0 else ''}{pnl:,.0f}฿ ({pnl_pct:+.2f}%)",
                    "inline": True,
                },
                {
                    "name": "📊 VIX",
                    "value": f"{ps.get('vix', 0):.1f}",
                    "inline": True,
                },
                {
                    "name": "🏆 ตัวเด่น",
                    "value": f"{top_g.get('ticker', '-')} (+{top_g.get('pct', 0):.1f}%)",
                    "inline": True,
                },
                {
                    "name": "😢 ตัวร่วง",
                    "value": f"{top_l.get('ticker', '-')} ({top_l.get('pct', 0):.1f}%)",
                    "inline": True,
                },
                {
                    "name": "🌍 Regime",
                    "value": ps.get("regime", "?"),
                    "inline": True,
                },
            ],
            "footer": {
                "text": f"QuantHQ — ถือ {ps.get('holdings_count', 0)} ตัว"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return self._send({"embeds": [embed]})

    # ==========================================
    # 🛑 5. Circuit Breaker Alert
    # ==========================================
    def send_circuit_breaker(self, reason, drawdown_pct, action_taken):
        """
        แจ้งเตือนเมื่อ Circuit Breaker ถูกกระตุ้น

        Args:
            reason: เหตุผลที่ trigger เช่น "Portfolio Drawdown > 25%"
            drawdown_pct: drawdown ปัจจุบัน (%)
            action_taken: สิ่งที่ระบบทำ เช่น "หยุดคำสั่งซื้อทั้งหมด"
        """
        embed = {
            "title": "🛑 CIRCUIT BREAKER ACTIVATED",
            "description": f"**ระบบหยุดทำงานอัตโนมัติ**\n\n{reason}",
            "color": 0xE74C3C,
            "fields": [
                {
                    "name": "📉 Drawdown",
                    "value": f"{drawdown_pct:.1f}%",
                    "inline": True,
                },
                {
                    "name": "⚡ Action",
                    "value": action_taken,
                    "inline": True,
                },
                {
                    "name": "🔧 วิธีแก้",
                    "value": "1. ตรวจสอบสถานการณ์ตลาด\n2. ทบทวนกลยุทธ์\n3. Reset circuit breaker เมื่อพร้อม",
                    "inline": False,
                },
            ],
            "footer": {"text": "QuantHQ Risk Engine — Manual Review Required"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return self._send({"content": "🛑 **@everyone** — Circuit Breaker ถูกกระตุ้น!", "embeds": [embed]})

    # ==========================================
    # 🎯 6. New Position Signal
    # ==========================================
    def send_new_position_signal(self, ticker, alpha_score, sector, reasons):
        """
        แจ้งเตือนเมื่อพบหุ้นใหม่ที่น่าสนใจ

        Args:
            ticker: ชื่อหุ้น เช่น "LLY"
            alpha_score: คะแนน Alpha
            sector: sector ของหุ้น
            reasons: list ของเหตุผล
        """
        reasons_text = "\n".join([f"✅ {r}" for r in reasons])

        embed = {
            "title": f"🎯 New Position Signal: {ticker}",
            "description": f"ระบบตรวจพบหุ้นใหม่ที่ผ่านเกณฑ์คัดกรอง",
            "color": 0x9B59B6,
            "fields": [
                {"name": "📊 Alpha Score", "value": f"{alpha_score:.2f}", "inline": True},
                {"name": "🏭 Sector", "value": sector, "inline": True},
                {"name": "📋 เหตุผล", "value": reasons_text[:1024], "inline": False},
            ],
            "footer": {"text": "QuantHQ Alpha Radar"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return self._send({"embeds": [embed]})

    # ==========================================
    # 🏥 7. Factor Health Report
    # ==========================================
    def send_factor_health(self, factor_report):
        """
        ส่งรายงานสุขภาพ Factor ประจำสัปดาห์

        Args:
            factor_report: dict เช่น {
                "Momentum": {"ic": 0.08, "icir": 0.45, "status": "HEALTHY"},
                "Quality": {"ic": 0.03, "icir": 0.15, "status": "WEAK"},
                "Value": {"ic": -0.02, "icir": -0.10, "status": "BROKEN"},
            }
        """
        status_emoji = {"HEALTHY": "🟢", "WEAK": "🟡", "BROKEN": "🔴"}

        lines = ""
        for factor_name, data in factor_report.items():
            emoji = status_emoji.get(data.get("status", ""), "❓")
            lines += f"{emoji} **{factor_name}**: IC={data.get('ic', 0):.3f} | ICIR={data.get('icir', 0):.2f}\n"

        any_broken = any(d.get("status") == "BROKEN" for d in factor_report.values())

        embed = {
            "title": "🏥 Factor Health Report",
            "description": lines,
            "color": 0xE74C3C if any_broken else 0x2ECC71,
            "footer": {"text": "QuantHQ Alpha Research Lab"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        content = ""
        if any_broken:
            content = "⚠️ **Factor บางตัวกลับทิศ — ตรวจสอบด่วน!**"

        return self._send({"content": content, "embeds": [embed]})


    # ==========================================
    # 📰 สรุปข่าวสาร AI Sentiment (News Alert)
    # ==========================================
    def send_news_alert(self, ticker, sentiment, impact, summary, url):
        """ส่งแจ้งเตือนข่าวและ AI Sentiment"""
        color = 0x2ECC71 if sentiment == "Bullish" else 0xE74C3C if sentiment == "Bearish" else 0x95A5A6
        emoji = "🚀" if sentiment == "Bullish" else "🩸" if sentiment == "Bearish" else "😐"
        
        embed = {
            "title": f"{emoji} AI News Alert: {ticker}",
            "description": f"**Sentiment:** {sentiment} | **Impact:** {impact}/10\n\n**AI Summary:**\n{summary}\n\n[อ่านข่าวต้นฉบับ]({url})",
            "color": color,
            "footer": {"text": "QuantHQ AI Sentiment System"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        content = f"**{ticker}** มีข่าวสำคัญ! (Impact: {impact}/10)" if impact >= 7 else ""
        return self._send({"content": content, "embeds": [embed]})

# ==========================================
# 🧪 ฟังก์ชันทดสอบ
# ==========================================
def test_discord_connection(webhook_url):
    """ทดสอบว่า Webhook ใช้งานได้"""
    notifier = DiscordNotifier(webhook_url)
    success = notifier._send({
        "content": "✅ QuantHQ Terminal เชื่อมต่อ Discord สำเร็จ!",
        "embeds": [{
            "title": "🔔 Connection Test",
            "description": "ระบบแจ้งเตือนพร้อมใช้งาน",
            "color": 0x2ECC71,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    })
    return success
