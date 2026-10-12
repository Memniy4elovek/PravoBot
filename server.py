
import certifi
import ssl
import json
import os
import time
import uuid
import socket
import urllib.request
import urllib.error
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Настройки GigaChat
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra")

TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://api.giga.chat/v1/chat/completions"

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

app = FastAPI(title="PravoBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


SYSTEM_PROMPT = """
Ты — ПравоБот, помощник по первичной правовой информации
в Российской Федерации.

Правила ответа:
1. Всегда отвечай на русском языке.
2. Сразу отвечай по существу, без рассуждений вслух.
3. Используй понятную структуру и короткие абзацы.
4. Не выдумывай статьи законов, судебные решения и ссылки.
5. Если не уверен в норме права, прямо сообщи об этом.
6. Учитывай, что законодательство может меняться.
7. Если данных недостаточно, задай уточняющий вопрос.
8. Помогай составлять черновики правовых документов.
9. Не обещай гарантированный юридический результат.
10. Напоминай, что ответ не заменяет консультацию юриста.
"""

_access_token = None
_token_expires_at = 0


def get_access_token():
    global _access_token, _token_expires_at

    if not GIGACHAT_CREDENTIALS:
        raise HTTPException(
            status_code=500,
            detail="В Render не задан GIGACHAT_CREDENTIALS."
        )

    # Используем действующий токен, если он ещё не истёк
    if _access_token and time.time() < _token_expires_at - 60:
        return _access_token

    data = urlencode({
        "scope": "GIGACHAT_API_PERS"
    }).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={
            "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=20,
            context=SSL_CONTEXT
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

        _access_token = result.get("access_token")
        expires_at = result.get("expires_at")

        if not _access_token:
            print(
                "GigaChat OAuth: access_token отсутствует",
                flush=True
            )
            raise HTTPException(
                status_code=502,
                detail="GigaChat не вернул токен доступа."
            )

        if expires_at:
            expires_at = float(expires_at)

            # При необходимости преобразуем миллисекунды в секунды
            if expires_at > 10**12:
                expires_at /= 1000

            _token_expires_at = expires_at
        else:
            _token_expires_at = time.time() + 1800

        return _access_token

    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode(
            "utf-8",
            errors="replace"
        )
        print(
            f"GigaChat OAuth HTTP {exc.code}: {error_text[:500]}",
            flush=True
        )
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка авторизации GigaChat (HTTP {exc.code})."
        ) from exc

    except (
        urllib.error.URLError,
        socket.timeout,
        TimeoutError,
        OSError
    ) as exc:
        print(
            f"GigaChat OAuth connection error: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )
        raise HTTPException(
            status_code=504,
            detail="Не удалось подключиться к GigaChat для авторизации."
        ) from exc


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "PravoBot"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL
    }


@app.post("/chat")
def chat(request: ChatRequest):
    global _access_token, _token_expires_at

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Сообщение не должно быть пустым."
        )

    token = get_access_token()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message
            },
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": False,
    }

    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=90,
            context=SSL_CONTEXT
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        choices = result.get("choices", [])
        reply = ""

        if choices:
            content = choices[0].get(
                "message", {}
            ).get("content", "")

            if isinstance(content, str):
                reply = content.strip()

        if not reply:
            print(
                "GigaChat: ответ без текста",
                flush=True
            )
            raise HTTPException(
                status_code=502,
                detail="GigaChat не вернул текстовый ответ."
            )

        return {
            "reply": reply
        }

    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode(
            "utf-8",
            errors="replace"
        )
        print(
            f"GigaChat API HTTP {exc.code}: {error_text[:500]}",
            flush=True
        )

        if exc.code == 401:
            _access_token = None
            _token_expires_at = 0

            raise HTTPException(
                status_code=502,
                detail="GigaChat отклонил токен доступа."
            ) from exc

        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Превышен лимит запросов GigaChat."
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"Ошибка GigaChat API (HTTP {exc.code})."
        ) from exc

    except (
        urllib.error.URLError,
        socket.timeout,
        TimeoutError,
        OSError
    ) as exc:
        print(
            f"GigaChat API connection error: "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )
        raise HTTPException(
            status_code=504,
            detail="GigaChat не ответил вовремя. Попробуй ещё раз."
        ) from exc
