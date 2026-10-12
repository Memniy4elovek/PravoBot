
import json
import urllib.request
import urllib.error
import socket

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "pravobot-model"

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
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Сообщение не должно быть пустым."
        )

    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": 256,
            "temperature": 0.3
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.loads(response.read().decode("utf-8"))

        reply = result.get("message", {}).get("content", "").strip()

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="Модель не вернула текстовый ответ."
            )

        return {"reply": reply}

    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка Ollama ({exc.code}): {error_text}"
        ) from exc

    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama не ответила вовремя. Попробуй ещё раз."
        ) from exc
