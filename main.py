"""Secure & Stable WhatsApp AI Assistant (Anthropic Primary + Gemini Fallback + Green-API + GitHub Bridge)

- المحرك الأساسي للردود النصية: Anthropic REST API (موديل افتراضي اقتصادي قابل للتعديل عبر ANTHROPIC_MODEL).
- المحرك الاحتياطي التلقائي: Gemini REST API (للردود عند فشل/مهلة Anthropic، وللرسائل الصوتية).
- يرد على Green-API فوراً ويعالج الرسالة بالخلفية (يمنع التكرار والـ timeout).
- قفل أمان صارم: لا يرد ولا يرسل إلا لرقمك (MY_PHONE). إذا ما انضبط الرقم => يتجاهل كل شيء.
- جسر أوامر واتساب <-> GitHub Issues (نوفا / عبدالعزيز).
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("whatsapp-agent")

app = FastAPI(title="Secure WhatsApp Agent")

APP_VERSION = "6.0-anthropic-gemini-bridge"

# ---------------------------------------------------------------- الإعدادات
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5").strip() or "claude-haiku-4-5"
ANTHROPIC_FALLBACK_MODELS = ["claude-haiku-4-5", "claude-3-5-haiku-latest"]
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-flash-latest"]
_preferred_gemini_model: str | None = None

MY_PHONE = os.environ.get("MY_PHONE", "").strip().replace("+", "")
GREEN_ID = os.environ.get("GREEN_ID", "").strip()
GREEN_TOKEN = os.environ.get("GREEN_TOKEN", "").strip()
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip()  # اختياري

# جسر واتساب <-> GitHub
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "alialhmeed1234-sudo/skills").strip()
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "").strip()

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15MB
MAX_REPLY_CHARS = 3500
HISTORY_TURNS = 8  # عدد رسائل الذاكرة القصيرة لكل محادثة
ANTHROPIC_TIMEOUT = 15
GEMINI_TIMEOUT = 15
MAX_TOTAL_SECONDS = 45

SYSTEM_PROMPT = (
    "أنت مساعد تنفيذي ذكي ومساعد شخصي لعلي. "
    "أجب باللغة العربية باختصار واحترافية، وبنفس لغة المستخدم إذا كتب بلغة ثانية."
)

# ---------------------------------------------------------------- حالة داخلية
_seen_ids: "OrderedDict[str, float]" = OrderedDict()
_seen_lock = threading.Lock()
_history: dict[str, deque] = {}
_history_lock = threading.Lock()
_events: deque = deque(maxlen=30)
_last_engine: str = "none"


def _event(kind: str, detail: str = "") -> None:
    _events.append({
        "utc": time.strftime("%H:%M:%S", time.gmtime()),
        "event": kind,
        "detail": str(detail)[:140],
    })


def _already_processed(message_id: str) -> bool:
    """يمنع معالجة نفس الرسالة مرتين (Green-API قد يعيد الإرسال)."""
    if not message_id:
        return False
    with _seen_lock:
        if message_id in _seen_ids:
            return True
        _seen_ids[message_id] = time.time()
        while len(_seen_ids) > 500:
            _seen_ids.popitem(last=False)
    return False


def _get_history(chat_id: str) -> list[dict]:
    with _history_lock:
        return list(_history.get(chat_id, []))


def _remember(chat_id: str, user_text: str, bot_text: str) -> None:
    with _history_lock:
        dq = _history.setdefault(chat_id, deque(maxlen=HISTORY_TURNS))
        dq.append({"user": user_text, "assistant": bot_text})


def _build_anthropic_messages(chat_id: str, current_user_text: str) -> list[dict]:
    msgs: list[dict] = []
    for turn in _get_history(chat_id):
        msgs.append({"role": "user", "content": turn["user"]})
        msgs.append({"role": "assistant", "content": turn["assistant"]})
    msgs.append({"role": "user", "content": current_user_text})
    return msgs


def _build_gemini_contents(chat_id: str, parts: list[dict]) -> list[dict]:
    contents: list[dict] = []
    for turn in _get_history(chat_id):
        contents.append({"role": "user", "parts": [{"text": turn["user"]}]})
        contents.append({"role": "model", "parts": [{"text": turn["assistant"]}]})
    contents.append({"role": "user", "parts": parts})
    return contents


# ---------------------------------------------------------------- HTTP helpers
def http_post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 20) -> tuple[int, dict]:
    """POST JSON. يرجع (status, json). لا يطبع الـ URL أبداً (قد يحتوي على توكن)."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body[:300]}


def http_get_json(url: str, headers: dict | None = None, timeout: int = 15) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", errors="replace"))
        except Exception:
            return e.code, {}


def http_get_bytes(url: str, timeout: int = 30, limit: int = MAX_AUDIO_BYTES) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file too large")
    return data


# ---------------------------------------------------------------- واتساب
def send_whatsapp(chat_id: str, text: str) -> bool:
    if not GREEN_ID or not GREEN_TOKEN:
        logger.error("[GREEN-API] Credentials missing")
        _event("send_fail", "green credentials missing")
        return False

    # 🔒 قفل الأمان: لا إرسال إلا لرقمك الشخصي
    clean = chat_id.replace("@c.us", "").replace("@g.us", "")
    if not MY_PHONE or clean != MY_PHONE:
        logger.warning("[SECURITY BLOCK] blocked send to unauthorized chat")
        return False

    url = f"https://api.green-api.com/waInstance{GREEN_ID}/sendMessage/{GREEN_TOKEN}"
    try:
        status, _ = http_post_json(url, {"chatId": chat_id, "message": text[:MAX_REPLY_CHARS]}, timeout=20)
        logger.info("[WHATSAPP SENT] status=%s", status)
        _event("send", f"status={status}")
        return 200 <= status < 300
    except Exception as e:
        logger.error("[WHATSAPP SEND ERROR] %s", type(e).__name__)
        _event("send_fail", type(e).__name__)
        return False


# ---------------------------------------------------------------- Anthropic (المحرك الأساسي)
def _anthropic_model_candidates() -> list[str]:
    seen, out = set(), []
    for m in [ANTHROPIC_MODEL, *ANTHROPIC_FALLBACK_MODELS]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _extract_anthropic_text(resp: dict) -> str:
    blocks = resp.get("content") or []
    return "".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def ask_anthropic(chat_id: str, user_text: str, deadline: float) -> tuple[str | None, str]:
    """يرجع (نص_الرد_أو_None, وصف_آخر_خطأ)."""
    if not ANTHROPIC_API_KEY:
        return None, "ANTHROPIC_API_KEY missing"

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    messages = _build_anthropic_messages(chat_id, user_text)
    last_err = "unknown"

    for model in _anthropic_model_candidates():
        if time.time() >= deadline:
            break
        payload = {
            "model": model,
            "max_tokens": 1200,
            "temperature": 0.5,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        for attempt in range(2):
            remaining = deadline - time.time()
            if remaining <= 1.0:
                break
            req_timeout = max(3, min(ANTHROPIC_TIMEOUT, int(remaining)))
            try:
                status, body = http_post_json(ANTHROPIC_URL, payload, headers=headers, timeout=req_timeout)
            except Exception as e:
                last_err = f"{model}: network-{type(e).__name__}"
                logger.warning("[ANTHROPIC] %s", last_err)
                time.sleep(1.2 * (attempt + 1))
                continue

            if status == 200:
                text = _extract_anthropic_text(body)
                if text:
                    _event("anthropic_ok", model)
                    return text, ""
                last_err = f"{model}: empty response"
                break

            msg = ((body.get("error") or {}).get("message", "") if isinstance(body, dict) else "")[:120]
            last_err = f"{model}: HTTP {status} {msg}"
            logger.warning("[ANTHROPIC] %s", last_err)

            if status in (401, 403):
                # مفتاح غير صالح أو منتهي الرصيد -> انتقل فوراً للـ fallback بدون إضاعة وقت
                return None, last_err
            if status in (429, 500, 502, 503, 529):
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    return None, last_err


# ---------------------------------------------------------------- Gemini (الاحتياطي + الصوت)
def _extract_gemini_text(resp: dict) -> str:
    cands = resp.get("candidates") or []
    if not cands:
        block = (resp.get("promptFeedback") or {}).get("blockReason")
        return "ما قدرت أرد على هذي الرسالة (تم حظرها من النموذج)." if block else ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def _gemini_model_candidates() -> list[str]:
    seen, out = set(), []
    for m in [_preferred_gemini_model, GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


_SUGGEST_RE = re.compile(r"use models/([A-Za-z0-9._\-]+)")
_disc_cache: dict = {"t": 0.0, "models": []}
_SKIP_WORDS = ("image", "tts", "live", "audio", "embedding", "vision", "imagen", "veo", "gemma", "robotics", "computer")


def _discover_gemini_models() -> list[str]:
    if not GEMINI_API_KEY:
        return []
    now = time.time()
    if _disc_cache["models"] and now - _disc_cache["t"] < 3600:
        return list(_disc_cache["models"])
    try:
        status, body = http_get_json(f"{GEMINI_BASE}?pageSize=200", headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=12)
    except Exception:
        return []
    if status != 200:
        return []
    names = []
    for m in body.get("models", []):
        n = (m.get("name") or "").replace("models/", "")
        if (
            "generateContent" in (m.get("supportedGenerationMethods") or [])
            and "flash" in n
            and not any(w in n for w in _SKIP_WORDS)
        ):
            names.append(n)
    names.sort(reverse=True)
    _disc_cache.update(t=now, models=names)
    return list(names)


def ask_gemini_fallback(
    chat_id: str,
    user_text: str,
    deadline: float,
    audio_bytes: bytes | None = None,
    audio_mime: str = "audio/ogg",
) -> tuple[str | None, str]:
    """يرجع (نص_الرد_أو_None, وصف_آخر_خطأ)."""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY missing"

    if audio_bytes:
        parts: list[dict] = [
            {"inline_data": {"mime_type": audio_mime, "data": base64.b64encode(audio_bytes).decode("ascii")}},
            {"text": f"{SYSTEM_PROMPT}\n\nاستمع لهذه الرسالة الصوتية وأجب عليها بدقة واختصار."},
        ]
    else:
        parts = [{"text": f"{SYSTEM_PROMPT}\n\n{user_text}"}]

    payload = {
        "contents": _build_gemini_contents(chat_id, parts),
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1500},
    }
    headers = {"x-goog-api-key": GEMINI_API_KEY}

    global _preferred_gemini_model
    last_error = "unknown"
    models = _gemini_model_candidates()
    i = 0

    while i < len(models) and time.time() < deadline:
        model = models[i]
        i += 1
        url = f"{GEMINI_BASE}/{model}:generateContent"
        for attempt in range(2):
            remaining = deadline - time.time()
            if remaining <= 1.0:
                break
            req_timeout = max(3, min(GEMINI_TIMEOUT, int(remaining)))
            try:
                status, body = http_post_json(url, payload, headers=headers, timeout=req_timeout)
            except Exception as e:
                last_error = f"{model}: network-{type(e).__name__}"
                logger.warning("[GEMINI-FALLBACK] %s", last_error)
                time.sleep(1.0 * (attempt + 1))
                continue

            if status == 200:
                text = _extract_gemini_text(body)
                if text:
                    _event("gemini_ok", model)
                    _preferred_gemini_model = model
                    return text, ""
                last_error = f"{model}: empty response"
                break

            msg = ((body.get("error") or {}).get("message", "") if isinstance(body, dict) else "")[:120]
            last_error = f"{model}: HTTP {status} {msg}"
            logger.warning("[GEMINI-FALLBACK] %s", last_error)

            if status == 404:
                m = _SUGGEST_RE.search(msg)
                if m and m.group(1) not in models:
                    models.append(m.group(1))
                break
            if status in (401, 403):
                return None, last_error
            if status in (429, 500, 502, 503, 504):
                time.sleep(1.2 * (attempt + 1))
                continue
            break

    tried = set(models)
    extra_n = 0
    for name in _discover_gemini_models():
        remaining = deadline - time.time()
        if name in tried or extra_n >= 2 or remaining <= 2.0:
            continue
        extra_n += 1
        try:
            status, body = http_post_json(
                f"{GEMINI_BASE}/{name}:generateContent",
                payload,
                headers=headers,
                timeout=max(3, min(GEMINI_TIMEOUT, int(remaining))),
            )
        except Exception as e:
            last_error = f"{name}: network-{type(e).__name__}"
            continue
        if status == 200:
            text = _extract_gemini_text(body)
            if text:
                _event("gemini_ok", name)
                _preferred_gemini_model = name
                return text, ""
        last_error = f"{name}: HTTP {status}"

    return None, last_error


def generate_reply(
    chat_id: str,
    user_text: str,
    audio_bytes: bytes | None = None,
    audio_mime: str = "audio/ogg",
) -> str:
    """يشغّل Anthropic كمحرك أساسي للنص، وينتقل تلقائياً إلى Gemini عند الفشل أو للرسائل الصوتية."""
    global _last_engine
    deadline = time.time() + MAX_TOTAL_SECONDS
    memory_label = "[رسالة صوتية]" if audio_bytes else user_text

    # للرسائل الصوتية نستخدم Gemini مباشرة ليدعم الصوت؛ وللنص نبدأ بـ Anthropic
    if not audio_bytes and user_text:
        text, anth_err = ask_anthropic(chat_id, user_text, deadline)
        if text:
            _last_engine = "anthropic"
            _remember(chat_id, memory_label, text)
            return text
        _event("fallback_to_gemini", anth_err[:90])

    prompt_text = user_text or "لخّص الرسالة الصوتية وأجب عليها."
    gem_text, gem_err = ask_gemini_fallback(
        chat_id,
        prompt_text,
        deadline,
        audio_bytes=audio_bytes,
        audio_mime=audio_mime,
    )
    if gem_text:
        _last_engine = "gemini"
        _remember(chat_id, memory_label, gem_text)
        return gem_text

    _last_engine = "none"
    detail = gem_err if audio_bytes else f"anth=({anth_err[:50]}) | gem=({gem_err[:50]})"
    logger.error("[AI FAILED] %s", detail)
    _event("ai_fail", detail[:120])
    return "حصل خطأ مؤقت في المعالجة حالياً، جرّب ترسل رسالتك مرة ثانية بعد شوي."


# ---------------------------------------------------------------- جسر الأوامر: واتساب -> GitHub Issue
_CMD_RE = re.compile(r"^\s*(عبد\s?العزيز|نوفا)\s*[:：،\-]\s*(.+)$", re.S)
_MENTION = {"عبدالعزيز": "@claude", "نوفا": "@nova"}
MAX_COMMANDS_PER_DAY = 20
_cmd_day: dict = {"d": "", "n": 0}


def _cmd_allowed() -> bool:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if _cmd_day["d"] != today:
        _cmd_day.update(d=today, n=0)
    if _cmd_day["n"] >= MAX_COMMANDS_PER_DAY:
        return False
    _cmd_day["n"] += 1
    return True


def parse_command(text: str):
    m = _CMD_RE.match(text or "")
    if not m:
        return None
    agent = m.group(1).replace(" ", "")
    return agent, m.group(2).strip()


def handle_command(chat_id: str, agent: str, task: str) -> None:
    if not GITHUB_TOKEN:
        send_whatsapp(chat_id, "🤖 ما أقدر أرسل الأمر: GITHUB_TOKEN غير مضاف في البيئة.")
        _event("cmd_fail", "no github token")
        return
    if not _cmd_allowed():
        send_whatsapp(chat_id, f"🤖 وصلت الحد اليومي للأوامر ({MAX_COMMANDS_PER_DAY}). نكمل بكرة.")
        _event("cmd_fail", "daily cap")
        return
    title = "أمر واتساب: " + task.replace("\n", " ")[:60]
    body = f"{_MENTION[agent]} {task}\n\n(مصدر الأمر: واتساب)"
    try:
        status, resp = http_post_json(
            f"https://api.github.com/repos/{GITHUB_REPO}/issues",
            {"title": title, "body": body},
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "nabras-bridge",
            },
            timeout=20,
        )
    except Exception as e:
        _event("cmd_fail", type(e).__name__)
        send_whatsapp(chat_id, "🤖 تعذّر إرسال الأمر الحين، جرّب بعد شوي.")
        return
    if status == 201 and isinstance(resp, dict):
        _event("cmd_sent", f"issue #{resp.get('number')}")
        send_whatsapp(chat_id, f"🤖 وصل أمرك لـ{agent} (طلب رقم {resp.get('number')}). بيجيك الرد هنا لما يخلص.")
    else:
        _event("cmd_fail", f"github status {status}")
        send_whatsapp(chat_id, f"🤖 GitHub رفض الأمر (رمز {status}). تأكد من صلاحيات التوكن واسم الريبو.")


def process_message(chat_id: str, user_text: str, audio_url: str, audio_mime: str) -> None:
    try:
        audio_bytes = None
        if audio_url:
            try:
                audio_bytes = http_get_bytes(audio_url)
            except Exception as e:
                logger.error("[AUDIO] download failed: %s", type(e).__name__)
                send_whatsapp(chat_id, "🤖 ما قدرت أنزّل الرسالة الصوتية، جرّب مرة ثانية.")
                return
        if not user_text and not audio_bytes:
            return
        cmd = parse_command(user_text) if user_text else None
        if cmd:
            handle_command(chat_id, cmd[0], cmd[1])
            return
        reply = generate_reply(chat_id, user_text, audio_bytes=audio_bytes, audio_mime=audio_mime)
        send_whatsapp(chat_id, f"🤖 {reply}")
    except Exception as e:
        logger.exception("[PROCESS ERROR] %s", type(e).__name__)


# ---------------------------------------------------------------- المسارات
@app.get("/")
def home() -> dict[str, str]:
    return {"status": "active", "agent": "Secure Agent Active", "version": APP_VERSION}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "primary_engine": "anthropic",
        "fallback_engine": "gemini",
        "last_engine": _last_engine,
        "anthropic_model": ANTHROPIC_MODEL,
        "gemini_model": GEMINI_MODEL,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "green_api_set": bool(GREEN_ID and GREEN_TOKEN),
        "my_phone_set": bool(MY_PHONE),
        "github_bridge": bool(GITHUB_TOKEN and GITHUB_WEBHOOK_SECRET),
    }


_diag_cache: dict = {"t": 0.0, "data": None}


@app.get("/diag")
def diag() -> dict:
    """يفحص حالة المحرك الأساسي (Anthropic) والاحتياطي (Gemini) بدون كشف أي مفاتيح. مخزّن 60 ثانية."""
    now = time.time()
    if _diag_cache["data"] and now - _diag_cache["t"] < 60:
        return _diag_cache["data"]

    result: dict = {
        "version": APP_VERSION,
        "anthropic": {},
        "gemini_fallback": {"models": {}},
    }

    if not ANTHROPIC_API_KEY:
        result["anthropic"]["error"] = "ANTHROPIC_API_KEY missing"
    else:
        probe = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "ping"}],
        }
        try:
            st, body = http_post_json(
                ANTHROPIC_URL,
                probe,
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": ANTHROPIC_VERSION},
                timeout=12,
            )
            msg = ((body.get("error") or {}).get("message", "") if isinstance(body, dict) else "")[:160]
            result["anthropic"] = {"model": ANTHROPIC_MODEL, "status": st, "error": msg}
        except Exception as e:
            result["anthropic"] = {"model": ANTHROPIC_MODEL, "status": "network", "error": type(e).__name__}

    if not GEMINI_API_KEY:
        result["gemini_fallback"]["error"] = "GEMINI_API_KEY missing"
    else:
        g_probe = {"contents": [{"role": "user", "parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 16}}
        for m in _gemini_model_candidates():
            try:
                status, body = http_post_json(
                    f"{GEMINI_BASE}/{m}:generateContent",
                    g_probe,
                    headers={"x-goog-api-key": GEMINI_API_KEY},
                    timeout=12,
                )
                msg = ((body.get("error") or {}).get("message", "") if isinstance(body, dict) else "")[:160]
                result["gemini_fallback"]["models"][m] = {"status": status, "error": msg}
            except Exception as e:
                result["gemini_fallback"]["models"][m] = {"status": "network", "error": type(e).__name__}
        result["gemini_fallback"]["available_flash_models"] = _discover_gemini_models()[:10]

    _diag_cache.update(t=now, data=result)
    return result


@app.post("/github-webhook")
async def github_webhook(request: Request, background: BackgroundTasks):
    """يستقبل ردود الوكلاء من GitHub (تعليقات على Issues) ويرسلها لواتساب علي فقط."""
    if not GITHUB_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="not configured")
    raw = await request.body()
    sig = request.headers.get("x-hub-signature-256", "")
    expected = "sha256=" + hmac.new(GITHUB_WEBHOOK_SECRET.encode(), raw, "sha256").hexdigest()
    if not hmac.compare_digest(sig, expected):
        _event("gh_rejected", "bad signature")
        raise HTTPException(status_code=401, detail="bad signature")
    ev = request.headers.get("x-github-event", "")
    if ev == "ping":
        return {"status": "pong"}
    try:
        d = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if ev != "issue_comment" or d.get("action") != "created":
        return {"status": "ignored"}
    c, i = d.get("comment") or {}, d.get("issue") or {}
    if (d.get("repository") or {}).get("full_name") != GITHUB_REPO:
        return {"status": "ignored", "reason": "other_repo"}
    if _already_processed("gh-" + str(c.get("id", ""))):
        return {"status": "ignored", "reason": "duplicate"}
    if (c.get("user") or {}).get("login") != "github-actions[bot]" or not str(i.get("title", "")).startswith("أمر واتساب"):
        return {"status": "ignored", "reason": "not_bridge_reply"}
    msg = f"📬 رد على أمرك (طلب {i.get('number')}):\n\n{str(c.get('body', ''))[:1500]}\n\nالتفاصيل: {c.get('html_url', '')}"
    _event("gh_reply", f"issue #{i.get('number')}")
    if MY_PHONE:
        background.add_task(send_whatsapp, f"{MY_PHONE}@c.us", msg)
    return {"status": "relayed"}


@app.get("/events")
def events() -> dict:
    """آخر 30 حدث (توقيت UTC). بدون نصوص رسائل أو أرقام. للتشخيص فقط."""
    return {"version": APP_VERSION, "utc_now": time.strftime("%H:%M:%S", time.gmtime()), "events": list(_events)}


_green_cache: dict = {"t": 0.0, "data": None}


@app.get("/green")
def green_status() -> dict:
    """يفحص حالة Instance في Green-API من السيرفر (بدون كشف التوكن أو الرقم). مخزّن 30 ثانية."""
    now = time.time()
    if _green_cache["data"] and now - _green_cache["t"] < 30:
        return _green_cache["data"]
    if not GREEN_ID or not GREEN_TOKEN:
        return {"error": "GREEN_ID/GREEN_TOKEN missing"}
    base = f"https://api.green-api.com/waInstance{GREEN_ID}"
    out: dict = {"version": APP_VERSION}

    def call(method: str):
        try:
            return http_get_json(f"{base}/{method}/{GREEN_TOKEN}", timeout=15)
        except Exception as e:
            return None, {"error": type(e).__name__}

    st, body = call("getStateInstance")
    out["state"] = body.get("stateInstance", f"http {st}") if isinstance(body, dict) else f"http {st}"

    st, body = call("getSettings")
    if isinstance(body, dict) and st == 200:
        url = (body.get("webhookUrl") or "").split("?")[0]
        out["webhook_url"] = url
        for k in ("incomingWebhook", "outgoingWebhook", "outgoingMessageWebhook", "outgoingAPIMessageWebhook"):
            out[k] = body.get(k)
    else:
        out["settings"] = f"http {st}"

    for name in ("lastIncomingMessages", "lastOutgoingMessages"):
        st, body = call(f"{name}")
        out[name + "_count"] = len(body) if isinstance(body, list) else f"http {st}"
    _green_cache.update(t=now, data=out)
    return out


def _authorized(request: Request) -> bool:
    if not WEBHOOK_SECRET:
        return True
    given = request.query_params.get("secret", "")
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        given = given or auth[7:].strip()
    return hmac.compare_digest(given, WEBHOOK_SECRET)


@app.api_route("/webhook", methods=["GET", "POST"])
async def whatsapp_webhook(request: Request, background: BackgroundTasks):
    if request.method == "GET":
        challenge = request.query_params.get("hub.challenge") or request.query_params.get("challenge")
        if challenge is not None:
            return PlainTextResponse(challenge)
        return {"status": "ready", "provider": "Green-API"}

    if not _authorized(request):
        _event("rejected", "bad secret")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = json.loads((await request.body()).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    webhook_type = data.get("typeWebhook")
    _event("webhook", webhook_type)
    if webhook_type not in ("incomingMessageReceived", "outgoingMessageReceived"):
        return {"status": "ignored", "typeWebhook": webhook_type}

    sender_data = data.get("senderData") or {}
    message_data = data.get("messageData") or {}
    chat_id = sender_data.get("chatId", "") or data.get("chatId", "") or ""

    # 🔒 فشل مغلق: بدون MY_PHONE ما نرد على أحد، والرد فقط لمحادثة خاصة برقمك
    if not MY_PHONE or not chat_id.endswith("@c.us") or chat_id.replace("@c.us", "") != MY_PHONE:
        _event("ignored", "unauthorized")
        return {"status": "ignored", "reason": "unauthorized_sender"}

    msg_type = message_data.get("typeMessage")
    user_text = ""
    audio_url = ""
    audio_mime = "audio/ogg"

    if msg_type == "textMessage":
        user_text = (message_data.get("textMessageData") or {}).get("textMessage", "").strip()
    elif msg_type == "extendedTextMessage":
        user_text = (message_data.get("extendedTextMessageData") or {}).get("text", "").strip()
    elif msg_type == "audioMessage":
        fdata = message_data.get("fileMessageData") or {}
        audio_url = fdata.get("downloadUrl", "") or ""
        audio_mime = (fdata.get("mimeType") or "audio/ogg").split(";")[0].strip() or "audio/ogg"

    # منع الحلقة: تجاهل ردود البوت نفسه
    if user_text.startswith("🤖"):
        _event("ignored", "self_bot_reply")
        return {"status": "ignored", "reason": "self_bot_reply"}

    if not user_text and not audio_url:
        _event("ignored", f"empty_message type={msg_type}")
        return {"status": "ignored", "reason": "empty_message"}

    if _already_processed(data.get("idMessage", "")):
        _event("ignored", "duplicate")
        return {"status": "ignored", "reason": "duplicate"}

    _event("accepted", f"type={msg_type}")
    background.add_task(process_message, chat_id, user_text, audio_url, audio_mime)
    return {"status": "accepted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
