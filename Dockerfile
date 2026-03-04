FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для Pillow и torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Зависимости отдельным слоем для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY doll_server.py .

# Порт по умолчанию для HF Spaces и Render
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "doll_server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
