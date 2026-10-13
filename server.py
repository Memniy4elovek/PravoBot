import json
import logging
import os
import ssl
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request

import certifi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------------------------
# Настройки приложения
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pravobot")

app = FastAPI(title="PravoBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GIGACHAT_CREDENTIALS = os.getenv(
    "GIGACHAT_CREDENTIALS", ""
).strip()

MODEL = os.getenv(
    "GIGACHAT_MODEL", "GigaChat-3-Ultra"
)

TOKEN_URL = (
    "https://ngw.devices.sberbank.ru:9443"
    "/api/v2/oauth"
)

CHAT_URL = (
    "https://api.giga.chat/v1/chat/completions"
)

# Сертификат Минцифры, добавленный в репозиторий GitHub
GIGACHAT_CERT = os.path.join(
    BASE_DIR, "Russian_Trusted_Root_CA.cer"
)


# --------------------------------------------------
# SSL: стандартные сертификаты + сертификат Минцифры
# --------------------------------------------------

def create_ssl_context():
    context = ssl.create_default_context(
        cafile=certifi.where()
    )

    if not os.path.isfile(GIGACHAT_CERT):
        raise RuntimeError(
            "Не найден сертификат GigaChat: "
            + GIGACHAT_CERT
        )

    try:
        context.load_verify_locations(
            cafile=GIGACHAT_CERT
        )
    except (ssl.SSLError, OSError) as exc:
        raise RuntimeError(
            "Не удалось загрузить сертификат "
            "Russian_Trusted_Root_CA.cer. "
            "Проверь его формат PEM."
        ) from exc

    return context


SSL_CONTEXT = create_ssl_context()


# --------------------------------------------------
# Кэш OAuth-токена
# --------------------------------------------------

_access_token = None
_token_expires_at = 0


def get_access_token():
    global _access_token, _token_expires_at

    if not GIGACHAT_CREDENTIALS:
        raise RuntimeError(
            "Не задана переменная "
            "GIGACHAT_CREDENTIALS в Render."
        )

    # Используем кэш, если токен ещё действителен
    if (
        _access_token
        and time.time() < _token_expires_at
    ):
        return _access_token

    headers = {
        "Authorization": (
            f"Basic {GIGACHAT_CREDENTIALS}"
        ),
        "RqUID": str(uuid.uuid4()),
        "Content-Type": (
            "application/x-www-form-urlencoded"
        ),
        "Accept": "application/json",
    }

    body = urllib.parse.urlencode({
        "scope": "GIGACHAT_API_PERS"
    }).encode("utf-8")

    request = urllib.request.Request(
        TOKEN_URL,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=25,
            context=SSL_CONTEXT,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        _access_token = result["access_token"]

        # Обновляем токен заранее, до истечения срока
        expires_at = result.get("expires_at")

        if expires_at:
            # GigaChat обычно возвращает время в мс
            expires_at = float(expires_at)

            if expires_at > 10**12:
                expires_at /= 1000

            _token_expires_at = expires_at - 60
        else:
            _token_expires_at = time.time() + 1500

        logger.info("OAuth-токен GigaChat получен")
        return _access_token

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8", errors="replace"
        )

        logger.error(
            "GigaChat OAuth HTTP %s: %s",
            exc.code,
            error_body[:1000],
        )

        raise RuntimeError(
            f"Ошибка авторизации GigaChat: HTTP {exc.code}"
        ) from exc

    except (
        urllib.error.URLError,
        TimeoutError,
        ssl.SSLError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        logger.error(
            "GigaChat OAuth error: %s: %s",
            type(exc).__name__,
            str(exc),
        )

        raise RuntimeError(
            "Не удалось получить OAuth-токен GigaChat."
        ) from exc


# --------------------------------------------------
# Модели запросов и ответов
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str


# --------------------------------------------------
# Основной запрос к GigaChat
# --------------------------------------------------

def ask_gigachat(message: str) -> str:
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": MODEL,
   "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — PravoBot, профессиональный юридический консультант по российскому законодательству. "
                    "Твоя цель — помочь составить полноценный, готовый к подаче процессуальный документ, а не шаблон с пустыми местами.\n\n"
                    "СТРОГИЙ ПОРЯДОК РАБОТЫ В ДИАЛОГЕ:\n\n"
                    "ЭТАП 1: УТОЧНЯЮЩИЕ ВОПРОСЫ (ОБЯЗАТЕЛЬНЫЙ ЭТАП)\n"
                    "Если пользователь только начал диалог, нажал кнопку «Подготовить с ПравоБотом» или кратко описал проблему, "
                    "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО сразу генерировать тег <document>.\n"
                    "Сначала поприветствуй пользователя, коротко оцени ситуацию с точки зрения закона РФ и задай нумерованный список из 4–6 конкретных вопросов, разделенных на понятные пункты:\n"
                    "1. Данные заявителя: ФИО, адрес с индексом, контактный телефон.\n"
                    "2. Данные адресата/ответчика: точное наименование магазина/компании/работодателя или ФИО лица, их адрес или сайт.\n"
                    "3. Даты и факты: когда был заключен договор/куплен товар/начались нарушения, реквизиты чеков/договоров.\n"
                    "4. Суть проблемы и суммы: точная сумма уплаты, характер недостатка или нарушения, проводилась ли проверка качества.\n"
                    "5. Требование: возврат всей суммы, замена, устранение брака, выплата компенсации или неустойки.\n"
                    "В конце сообщения обязательно напиши: «Вы можете ответить на эти пункты кратко прямо здесь (например: 1. Иванов И.И., Москва... 2. ООО Ромашка...), и на основе ваших ответов я сразу сформирую полностью готовый документ.»\n\n"
                    "ЭТАП 2: ФОРМИРОВАНИЕ ДОКУМЕНТА\n"
                    "Только ПОСЛЕ ТОГО, как пользователь ответил на твои уточняющие вопросы (или если он прямо написал «составь из того, что есть»), ты формируешь финальный текст.\n"
                    "Впиши в текст все предоставленные пользователем данные, сформулируй юридически обоснованную позицию со статьями законов РФ (ЗоЗПП, ТК РФ, ГК РФ и др.), укажи точные расчеты, требования, приложения и место для подписи.\n"
                    "ОБЯЗАТЕЛЬНО оберни весь текст документа в теги:\n"
                    "<document title=\"Точное название документа\">\n"
                    "...весь текст документа от шапки до подписи...\n"
                    "</document>\n\n"
                    "После закрывающего тега </document> дай короткую пошаговую памятку: в скольких экземплярах распечатать, как вручить (лично под роспись или заказным письмом с описью) и в какой срок ждать ответ по закону."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        "temperature": 0.25,
        "max_tokens": 3000,
    }

    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=90,
            context=SSL_CONTEXT,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        return result["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8", errors="replace"
        )

        logger.error(
            "GigaChat API HTTP %s: %s",
            exc.code,
            error_body[:1500],
        )

        # Если токен отклонён, сбрасываем кэш
        if exc.code == 401:
            global _access_token, _token_expires_at
            _access_token = None
            _token_expires_at = 0

            raise RuntimeError(
                "GigaChat отклонил токен авторизации. "
                "Попробуй повторить запрос."
            ) from exc

        if exc.code == 429:
            raise RuntimeError(
                "Превышен лимит запросов GigaChat. "
                "Попробуй позже."
            ) from exc

        raise RuntimeError(
            f"Ошибка GigaChat API: HTTP {exc.code}"
        ) from exc

    except (
        urllib.error.URLError,
        TimeoutError,
        ssl.SSLError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        logger.error(
            "GigaChat API error: %s: %s",
            type(exc).__name__,
            str(exc),
        )

        raise RuntimeError(
            "Не удалось получить ответ от GigaChat."
        ) from exc


# --------------------------------------------------
# Маршруты FastAPI
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "PravoBot API",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Введите сообщение.",
        )

    if len(message) > 12000:
        raise HTTPException(
            status_code=400,
            detail="Сообщение слишком длинное.",
        )

    try:
        reply = ask_gigachat(message)

        return {"reply": reply}

    except RuntimeError as exc:
        logger.error("Ошибка /chat: %s", str(exc))

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "Непредвиденная ошибка обработки /chat"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Внутренняя ошибка сервера. "
                "Попробуй позже."
            ),
        )
