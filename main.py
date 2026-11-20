import os
import re
import json
import hmac
import hashlib
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import HTMLResponse
import anthropic
from google import genai
from google.genai import types

app = FastAPI(title="Nova & Abdulaziz Executive WhatsApp System")

MY_PHONE = re.sub(r"\D", "", os.getenv("MY_PHONE", ""))
GREEN_ID = os.getenv("GREEN_ID", "")
GREEN_TOKEN = os.getenv("GREEN_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "alialhmeed1234-sudo/skills")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SENDER_NAME = os.getenv("SENDER_NAME", "Nova - Executive Office")

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

ANTHROPIC_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"

MEMORY: dict[str, list[dict]] = {}
ACTIVITY_LOG: list[dict] = []
STATE = {"last_engine": "none", "messages_handled": 0, "emails_sent": 0, "tasks_created": 0}

NOVA_OWNER_PROMPT = (
    "أنتِ 'نوفا'، مديرة الأعمال التنفيذية ومسؤولة التسويق وخدمة العملاء لدى الأستاذ علي.\n"
    "- تتحدثين بلهجة لبنانية بيضاء راقية، دافئة، وعملية.\n"
    "- مهمتك تحويل أوامر علي إلى إنجاز فوري: صياغة خطط تسويق، عروض أسعار، ردود عملاء، وإرسال إيميلات رسمية.\n"
    "- إذا طلب علي إرسال إيميل وحدد المستلم والموضوع/المحتوى، أخرجي في نهاية ردك:\n"
    '```json\n{"action": "send_email", "to": "email@domain.com", "subject": "عنوان الإيميل", "body": "نص الإيميل"}\n```'
)

NOVA_CLIENT_PROMPT = (
    "أنتِ 'نوفا'، مديرة مكتب الأعمال ومسؤولة خدمة العملاء والتسويق لدى الأستاذ علي.\n"
    "- تردين على استفسارات العملاء بلهجة لبنانية محببة واحترافية.\n"
    "- رحبي بالعميل، أجيبي بوضوح واختصار، واجمعي بياناته لرفعها للأستاذ علي."
)

ABDULAZIZ_PROMPT = (
    "أنت 'عبدالعزيز'، المطور التقني ومسؤول المشاريع والإنتاجية لدى الأستاذ علي.\n"
    "- تتحدث بلهجة سعودية مهنية مختصرة وعملية.\n"
    "- مهمتك تقسيم المشاريع، متابعة الإنجاز، حل المشاكل التقنية، وإنشاء مهام برمجية.\n"
    "- إذا طلب علي إنشاء مهمة برمجية، أخرج في نهاية ردك:\n"
    '```json\n{"action": "create_issue", "title": "عنوان المهمة", "body": "تفاصيل المهمة"}\n```'
)


def log_activity(actor: str, sender: str, summary: str, status: str = "تم"):
    ACTIVITY_LOG.insert(0, {
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "actor": actor,
        "sender": sender,
        "summary": summary[:160],
        "status": status,
    })
    if len(ACTIVITY_LOG) > 40:
        ACTIVITY_LOG.pop()


def send_whatsapp(chat_id: str, text: str):
    if not (GREEN_ID and GREEN_TOKEN):
        return
    url = f"https://api.green-api.com/waInstance{GREEN_ID}/sendMessage/{GREEN_TOKEN}"
    try:
        httpx.post(url, json={"chatId": chat_id, "message": text}, timeout=20)
    except Exception as e:
        log_activity("System", chat_id, f"فشل إرسال واتساب: {e}", "خطأ")


def execute_send_email(to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    if not (SMTP_USER and SMTP_PASS):
        return False, "بيانات SMTP_USER و SMTP_PASS غير مضافة في Render بعد."
    try:
        msg = MIMEMultipart()
        msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        STATE["emails_sent"] += 1
        return True, f"تم إرسال الإيميل بنجاح إلى {to_addr}"
    except Exception as e:
        return False, f"تعذر إرسال الإيميل: {e}"


def execute_github_issue(title: str, body: str) -> tuple[bool, str]:
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN غير مضاف في Render."
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    try:
        r = httpx.post(url, headers=headers, json={"title": title, "body": body}, timeout=20)
        if r.status_code in (200, 201):
            issue_url = r.json().get("html_url", "")
            STATE["tasks_created"] += 1
            return True, f"تم فتح المهمة على GitHub: {issue_url}"
        return False, f"خطأ GitHub ({r.status_code}): {r.text[:120]}"
    except Exception as e:
        return False, f"تعذر الاتصال بـ GitHub: {e}"


def ask_ai(system_prompt: str, user_text: str, chat_id: str) -> str:
    history = MEMORY.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    if len(history) > 12:
        history[:] = history[-12:]

    reply = ""
    if claude_client:
        try:
            resp = claude_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=history,
            )
            reply = resp.content[0].text
            STATE["last_engine"] = "anthropic"
        except Exception:
            reply = ""

    if not reply and gemini_client:
        try:
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system_prompt}\n\nسجل المحادثة:\n{transcript}",
            )
            reply = resp.text or ""
            STATE["last_engine"] = "gemini"
        except Exception as e:
            reply = f"عذراً، واجهت ضغطاً تقنياً لحظياً: {e}"

    if not reply:
        reply = "لم يتم تفعيل مفاتيح الذكاء الاصطناعي بعد."

    history.append({"role": "assistant", "content": reply})
    return reply


def process_media_with_gemini(file_url: str, mime_type: str, caption: str, system_prompt: str) -> str:
    if not gemini_client:
        return "وصلني الملف/الصوت، لكن GEMINI_API_KEY غير مضاف بعد لمعالجة الصوتيات والصور."
    try:
        file_bytes = httpx.get(file_url, timeout=30).content
        prompt = caption or "استمع أو حلل هذا الملف ونفّذ المطلوب منه وفق شخصيتك."
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                f"{system_prompt}\n\nطلب المستخدم: {prompt}",
            ],
        )
        STATE["last_engine"] = "gemini-multimodal"
        return resp.text or "تم استلام الملف ومعالجته."
    except Exception as e:
        return f"تعذر تحليل الملف الصوتي/المرئي: {e}"


def handle_actions(raw_reply: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_reply, re.DOTALL)
    clean_reply = re.sub(r"```json\s*\{.*?\}\s*```", "", raw_reply, flags=re.DOTALL).strip()
    if not match:
        return clean_reply

    try:
        data = json.loads(match.group(1))
        action = data.get("action")
        if action == "send_email":
            ok, msg = execute_send_email(data["to"], data["subject"], data["body"])
            return f"{clean_reply}\n\n{'✅' if ok else '⚠️'} {msg}"
        if action == "create_issue":
            ok, msg = execute_github_issue(data["title"], data["body"])
            return f"{clean_reply}\n\n{'✅' if ok else '⚠️'} {msg}"
    except Exception as e:
        return f"{clean_reply}\n\n⚠️ خطأ أثناء تنفيذ الإجراء: {e}"
    return clean_reply


def route_persona(sender_phone: str, text: str) -> tuple[str, str, str]:
    is_owner = bool(MY_PHONE) and sender_phone.endswith(MY_PHONE[-9:])
    if not is_owner:
        return "نوفا (خدمة العملاء)", NOVA_CLIENT_PROMPT, text

    stripped = text.strip()
    if stripped.startswith(("عبدالعزيز:", "عبد العزيز:", "عبدالعزيز ")):
        clean = re.sub(r"^عبد\s?العزيز[:\s]*", "", stripped, count=1)
        return "عبدالعزيز", ABDULAZIZ_PROMPT, clean or stripped
    if stripped.startswith(("نوفا:", "نوفا ")):
        clean = re.sub(r"^نوفا[:\s]*", "", stripped, count=1)
        return "نوفا", NOVA_OWNER_PROMPT, clean or stripped

    dev_keywords = ("كود", "برمجة", "مهمة", "مشروع", "تطوير", "github", "مشكلة تقنية", "سيرفر", "إنتاجية")
    if any(k in stripped.lower() for k in dev_keywords):
        return "عبدالعزيز", ABDULAZIZ_PROMPT, stripped
    return "نوفا", NOVA_OWNER_PROMPT, stripped


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "7.0-nova-abdulaziz-executive",
        "team": {"operations_marketing": "نوفا", "dev_projects": "عبدالعزيز"},
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "green_api_set": bool(GREEN_ID and GREEN_TOKEN),
        "smtp_email_set": bool(SMTP_USER and SMTP_PASS),
        "github_bridge": bool(GITHUB_TOKEN),
        "stats": STATE,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    rows = "".join(
        f"<tr><td>{item['time']}</td><td><b>{item['actor']}</b></td><td>{item['sender']}</td><td>{item['summary']}</td><td>{item['status']}</td></tr>"
        for item in ACTIVITY_LOG
    ) or "<tr><td colspan='5'>لا توجد عمليات مسجلة بعد.</td></tr>"

    return f"""
    <html dir="rtl" lang="ar">
    <head>
      <meta charset="utf-8"><title>مكتب نوفا وعبدالعزيز التنفيذي</title>
      <style>
        body {{ font-family: Tahoma, Arial; background: #0f172a; color: #f8fafc; padding: 24px; }}
        .cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
        .card {{ background: #1e293b; padding: 16px 24px; border-radius: 12px; min-width: 180px; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 12px; overflow: hidden; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #334155; text-align: right; }}
        th {{ background: #334155; }}
      </style>
    </head>
    <body>
      <h2>لوحة متابعة فريق العمل (نوفا & عبدالعزيز)</h2>
      <div class="cards">
        <div class="card">الرسائل المعالجة: <b>{STATE['messages_handled']}</b></div>
        <div class="card">الإيميلات المرسلة: <b>{STATE['emails_sent']}</b></div>
        <div class="card">مهام GitHub: <b>{STATE['tasks_created']}</b></div>
        <div class="card">آخر محرك نشط: <b>{STATE['last_engine']}</b></div>
      </div>
      <table>
        <thead><tr><th>الوقت</th><th>المسؤول</th><th>الجهة</th><th>الملخص</th><th>الحالة</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </body>
    </html>
    """


@app.post("/webhook")
async def green_webhook(request: Request):
    payload = await request.json()
    if payload.get("typeWebhook") != "incomingMessageReceived":
        return {"ok": True}

    sender_data = payload.get("senderData", {})
    chat_id = sender_data.get("chatId", "")
    sender_phone = re.sub(r"\D", "", sender_data.get("sender", ""))
    msg_data = payload.get("messageData", {})
    msg_type = msg_data.get("typeMessage", "")

    STATE["messages_handled"] += 1

    if msg_type in ("textMessage", "extendedTextMessage"):
        text = (
            msg_data.get("textMessageData", {}).get("textMessage")
            or msg_data.get("extendedTextMessageData", {}).get("text")
            or ""
        )
        actor, prompt, clean_text = route_persona(sender_phone, text)
        raw_reply = ask_ai(prompt, clean_text, chat_id)
        final_reply = handle_actions(raw_reply)
        send_whatsapp(chat_id, f"*{actor}:*\n{final_reply}")
        log_activity(actor, sender_phone, clean_text)

    elif msg_type in ("audioMessage", "imageMessage"):
        file_data = msg_data.get("fileMessageData", {})
        download_url = file_data.get("downloadUrl", "")
        mime = file_data.get("mimeType") or ("audio/ogg" if msg_type == "audioMessage" else "image/jpeg")
        caption = file_data.get("caption", "")
        actor, prompt, _ = route_persona(sender_phone, caption or "نوفا:")
        reply = process_media_with_gemini(download_url, mime, caption, prompt)
        send_whatsapp(chat_id, f"*{actor}:*\n{reply}")
        log_activity(actor, sender_phone, f"[{msg_type}] {caption}")

    return {"ok": True}


@app.post("/github-webhook")
async def github_webhook(request: Request, x_hub_signature_256: Optional[str] = Header(None)):
    raw_body = await request.body()
    if GITHUB_WEBHOOK_SECRET and x_hub_signature_256:
        expected = "sha256=" + hmac.new(
            GITHUB_WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")

    event = await request.json()
    if event.get("action") == "created" and "comment" in event:
        issue_title = event["issue"]["title"]
        comment_body = event["comment"]["body"]
        author = event["comment"]["user"]["login"]
        if MY_PHONE:
            send_whatsapp(
                f"{MY_PHONE}@c.us",
                f"*عبدالعزيز (تحديث GitHub):*\nرد جديد من @{author} على مهمة *{issue_title}*:\n{comment_body}",
            )
        log_activity("عبدالعزيز (GitHub)", author, f"{issue_title}: {comment_body}")
    return {"ok": True}
