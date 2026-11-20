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

# --- متغيرات البيئة ---
MY_PHONE = re.sub(r"\D", "", os.getenv("MY_PHONE", ""))
GREEN_ID = os.getenv("GREEN_ID", "")
GREEN_TOKEN = os.getenv("GREEN_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "alialhmeed1234-sudo/skills")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")

# إعدادات الإيميل (اختياري لتفعيل الإرسال الفعلي)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SENDER_NAME = os.getenv("SENDER_NAME", "Nova - Executive Office")

# --- تهيئة المحركات ---
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

ANTHROPIC_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"

# --- ذاكرة محادثات وسجل عمليات ---
MEMORY: dict[str, list[dict]] = {}
ACTIVITY_LOG: list[dict] = []
STATE = {"last_engine": "none", "messages_handled": 0, "emails_sent": 0, "tasks_created": 0}

# --- تعريف الشخصيات ---
NOVA_OWNER_PROMPT = """أنتِ 'نوفا'، مديرة الأعمال التنفيذية ومسؤولة التسويق وخدمة العملاء لدى الأستاذ علي.
- تتحدثين بلهجة لبنانية بيضاء راقية، دافئة، وعملية (مثل: تكرم عينك، ولا يهمك، من عيوني، خلينا نرتبها).
- مهمتك تحويل أوامر علي إلى إنجاز فوري: صياغة خطط تسويق، عروض أسعار، ردود عملاء، وإرسال إيميلات رسمية.
- إذا طلب علي إرسال إيميل وحدد المستلم والموضوع/المحتوى، أخرجي في نهاية ردك كتلة JSON دقيقة بهذا الشكل فقط لتنفيذ الإرسال تلقائياً:
```json
{"action": "send_email", "to": "email@domain.com", "subject": "عنوان الإيميل", "body": "نص الإيميل الاحترافي كاملاً"}
