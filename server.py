
import os
import json
import time
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3.8-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:generateContent"
)

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


@app.get("/")
def home():
    return {"status": "ok", "service": "PravoBot"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "api_key_configured": bool(API_KEY),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Сообщение не должно быть пустым."
        )

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API-ключ Gemini не настроен в Render."
        )

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": message}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
        }
    }

    req = urllib.request.Request(
        GEMINI_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY,
        },
        method="POST",
    )

    # До 3 попыток при временной перегрузке Gemini
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            candidates = result.get("candidates", [])
            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
                if candidates else []
            )

            reply = "\n".join(
                part.get("text", "")
                for part in parts
                if part.get("text")
            ).strip()

            if not reply:
                raise HTTPException(
                    status_code=502,
                    detail="Gemini не вернула текстовый ответ."
                )

            return {"reply": reply}

        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode(
                "utf-8", errors="replace"
            )

            # Повторяем запрос при перегрузке или ограничении запросов
            if exc.code in (503, 429) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue

            if exc.code in (400, 401, 403):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Gemini отклонила запрос. Проверь API-ключ, "
                        "доступность модели и ограничения API."
                    )
                ) from exc

            if exc.code == 429:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Превышен лимит запросов Gemini. "
                        "Попробуй позже."
                    )
                ) from exc

            if exc.code == 503:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Gemini временно перегружена. "
                        "Попробуй отправить сообщение позже."
                    )
                ) from exc

            raise HTTPException(
                status_code=502,
                detail=(
                    f"Ошибка Gemini API ({exc.code}): "
                    f"{error_text[:500]}"
                )
            ) from exc

        except (urllib.error.URLError, TimeoutError) as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Gemini не ответила вовремя. "
                    "Попробуй ещё раз."
                )
            ) from exc

    raise HTTPException(
        status_code=503,
        detail="Gemini временно недоступна. Попробуй позже."
    )
