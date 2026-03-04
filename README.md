# Blythello

Blythe Doll Generator — преобразование фотографий в стиль куклы Blythe с помощью Stable Diffusion.

## 🚀 Быстрый старт

### Локально

```bash
# 1. Установи зависимости
pip install -r requirements.txt

# 2. Скопируй и заполни переменные окружения
cp .env.example .env
# Открой .env и заполни нужные значения

# 3. Запусти сервер
python doll_server.py
```

Сервер будет доступен на http://localhost:8000
Документация API: http://localhost:8000/docs

### Docker

```bash
docker build -t blythe-doll .
docker run -p 7860:7860 --env-file .env blythe-doll
```

### Hugging Face Spaces

1. Создай новый Space (Docker SDK)
2. Загрузи `doll_server.py`, `requirements.txt`, `Dockerfile`
3. Добавь секреты в Settings → Variables and Secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `ALLOWED_DOMAIN`

## 📡 API

### POST /generate

Параметры (multipart/form-data):
| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| file | файл | обязательный | JPEG или PNG до 10 МБ |
| style | string | blythe | blythe / anime / watercolor |
| steps | int | 25 | 10–50 |
| strength | float | 0.65 | 0.1–1.0 |
| guidance | float | 7.5 | 1.0–20.0 |

**Пример:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "file=@photo.jpg" \
  -F "style=blythe" \
  --output result.png
```

Если сервер занят — вернёт `202 Accepted` с `task_id`.
Проверяй статус: `GET /queue/{task_id}`
Скачай результат: `GET /result/{task_id}`

### GET /health
Статус сервера, устройство, uptime, загрузка очереди.

### GET /styles
Список доступных стилей.

## ⚙️ Переменные окружения

| Переменная | Обязательна | Описание |
|------------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | нет | Токен бота для уведомлений |
| `TELEGRAM_CHAT_ID` | нет | ID чата для уведомлений |
| `ALLOWED_DOMAIN` | нет | Домен для CORS (по умолчанию blythello.com) |
| `PORT` | нет | Порт сервера (по умолчанию 8000) |
| `HEARTBEAT_URL` | нет | URL для пинга чтобы Render не засыпал |

## 🔒 Безопасность

- **Никогда** не коммить `.env` или `blythe_config.json` в git
- Токены хранятся только в переменных окружения
- `.gitignore` уже настроен правильно

## 📦 Зависимости

- Python 3.11+
- FastAPI + Uvicorn
- PyTorch 2.3+
- Diffusers (Stable Diffusion)
- httpx (async HTTP)
