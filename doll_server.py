#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blythe Doll Generator Server
Full-featured FastAPI server for Stable Diffusion img2img with Blythe style
Optimized for free hosting (Hugging Face, Render, Colab)
Telegram notifications on failure (optional)

FIXES:
- Replaced requests with httpx for async Telegram/heartbeat calls
- Added background worker to properly process task queue
- Used contextlib.nullcontext instead of custom implementation
- Tokens loaded only from environment variables (never hardcoded)
- Fixed queue worker lifecycle tied to lifespan
- Fixed torch.autocast deprecation (device_type param)
- Improved error handling with bare except → except Exception
- Added RGB conversion before resize in prepare_image
- Fixed cache key to include strength and guidance (was missing in deployer version)
"""

import os
import io
import time
import logging
import hashlib
import asyncio
from contextlib import asynccontextmanager, nullcontext
from typing import Optional, Dict, Any
from collections import OrderedDict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# ==================== ТОКЕНЫ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ALLOWED_DOMAIN = os.environ.get("ALLOWED_DOMAIN", "blythello.com")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "Blythe Doll Generator")

# ==================== ИМПОРТЫ ====================
try:
    import torch
    import httpx
    import uvicorn
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from fastapi.concurrency import run_in_threadpool
    from PIL import Image, UnidentifiedImageError
    from diffusers import StableDiffusionImg2ImgPipeline
    from transformers import logging as hf_logging
    import psutil
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установи зависимости: pip install -r requirements.txt")
    raise

# ==================== КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ ====================
hf_logging.set_verbosity_error()
logging.getLogger("diffusers").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==================== КОНФИГУРАЦИЯ СЕРВЕРА ====================
class Config:
    # Модели
    PRIMARY_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    FALLBACK_MODELS = [
        "OFA-Sys/small-stable-diffusion-v0",
        "hf-internal-testing/tiny-stable-diffusion-pipe",
    ]

    # Лимиты
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    IMAGE_SIZE = 512
    CACHE_SIZE = 50

    # Параметры генерации
    DEFAULT_STEPS = 25
    DEFAULT_STRENGTH = 0.65
    DEFAULT_GUIDANCE = 7.5
    MIN_STEPS = 10
    MAX_STEPS = 50
    MIN_STRENGTH = 0.1
    MAX_STRENGTH = 1.0
    MIN_GUIDANCE = 1.0
    MAX_GUIDANCE = 20.0

    # Порт
    PORT = int(os.environ.get("PORT", 8000))

    # Таймауты
    FILE_READ_TIMEOUT = 30
    GENERATION_TIMEOUT = 300
    REQUEST_TIMEOUT = 300

    # Оптимизации
    ENABLE_ATTENTION_SLICING = True
    CPU_FALLBACK_STEPS = 15

    # Heartbeat (для Render/HF)
    HEARTBEAT_URL = os.environ.get("HEARTBEAT_URL", "")
    HEARTBEAT_INTERVAL = 240  # 4 минуты


# ==================== СТИЛИ ====================
STYLES = {
    "blythe": {
        "positive": (
            "a beautiful custom Blythe doll, big glossy oversized eyes, "
            "kawaii chibi style, big head small body, pastel colors, "
            "soft watercolor illustration, detailed face painting, "
            "high quality art, adorable expression, studio lighting, masterpiece"
        ),
        "negative": (
            "realistic human photograph, real person, 3d render, "
            "ugly, distorted, blurry, low quality, horror, scary, deformed, nsfw"
        ),
    },
    "anime": {
        "positive": (
            "anime doll character, large sparkling eyes, cute chibi proportions, "
            "vibrant colors, clean line art, high quality illustration, kawaii style"
        ),
        "negative": "realistic, photograph, 3d, ugly, blurry, low quality",
    },
    "watercolor": {
        "positive": (
            "watercolor doll portrait, soft pastel tones, artistic brush strokes, "
            "dreamy aesthetic, Blythe doll inspired, high quality painting"
        ),
        "negative": "digital harsh, 3d render, photograph, ugly, low quality",
    },
}


# ==================== МОДЕЛИ ДАННЫХ ====================
class GenerationStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationTask:
    id: str
    status: GenerationStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result_bytes: Optional[bytes] = None  # храним результат в памяти


# ==================== TELEGRAM УВЕДОМЛЕНИЯ ====================
async def send_telegram_message(message: str) -> bool:
    """Отправка сообщения в Telegram через httpx (правильный async)"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🤖 {PROJECT_NAME}:\n{message}",
            "parse_mode": "HTML",
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=payload)
        return True
    except Exception as e:
        logger.debug(f"Telegram error (non-critical): {e}")
        return False


# ==================== LRU КЭШ ====================
class ResultCache:
    """LRU кэш результатов генерации"""

    def __init__(self, capacity: int = Config.CACHE_SIZE):
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[bytes]:
        if key in self.cache:
            self.cache.move_to_end(key)
            logger.debug(f"Cache HIT: {key[:8]}...")
            return self.cache[key]
        logger.debug(f"Cache MISS: {key[:8]}...")
        return None

    def set(self, key: str, value: bytes) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


# ==================== ОЧЕРЕДЬ ЗАДАЧ ====================
class TaskQueue:
    """Асинхронная очередь задач с воркером"""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, GenerationTask] = {}
        self.active_count: int = 0
        self._lock = asyncio.Lock()

    async def enqueue(self, task_id: str, image: Image.Image, params: dict) -> int:
        """Добавить задачу. Возвращает позицию в очереди (0 = сразу в работу)."""
        position = self.queue.qsize() + self.active_count
        task = GenerationTask(
            id=task_id,
            status=GenerationStatus.QUEUED,
            created_at=time.time(),
        )
        self.tasks[task_id] = task
        await self.queue.put((task_id, image, params))
        return position

    def get_status(self, task_id: str) -> Optional[GenerationTask]:
        return self.tasks.get(task_id)

    async def _mark_processing(self, task_id: str) -> None:
        async with self._lock:
            self.active_count += 1
            if task_id in self.tasks:
                self.tasks[task_id].status = GenerationStatus.PROCESSING
                self.tasks[task_id].started_at = time.time()

    async def _mark_done(self, task_id: str, result: Optional[bytes], error: Optional[str]) -> None:
        async with self._lock:
            self.active_count = max(0, self.active_count - 1)
            if task_id in self.tasks:
                if error:
                    self.tasks[task_id].status = GenerationStatus.FAILED
                    self.tasks[task_id].error = error
                else:
                    self.tasks[task_id].status = GenerationStatus.COMPLETED
                    self.tasks[task_id].completed_at = time.time()
                    self.tasks[task_id].result_bytes = result

    async def worker(self, pipeline_state: dict) -> None:
        """Фоновый воркер — читает очередь и выполняет генерацию"""
        logger.info("🔄 Queue worker запущен")
        while True:
            try:
                task_id, image, params = await self.queue.get()
                await self._mark_processing(task_id)
                try:
                    result = await _generate(pipeline_state, image, **params)
                    await self._mark_done(task_id, result, None)
                except Exception as exc:
                    logger.error(f"Worker generation error [{task_id}]: {exc}")
                    await self._mark_done(task_id, None, str(exc))
                    await send_telegram_message(f"❌ Ошибка генерации: {str(exc)[:200]}")
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                logger.info("🛑 Queue worker остановлен")
                break
            except Exception as exc:
                logger.error(f"Worker unexpected error: {exc}")
                await asyncio.sleep(1)


# ==================== СОСТОЯНИЕ ПРИЛОЖЕНИЯ ====================
pipeline_state: Dict[str, Any] = {
    "pipe": None,
    "device": "cpu",
    "dtype": torch.float32,
    "model_id": None,
    "ready": False,
    "start_time": time.time(),
}

cache = ResultCache()
task_queue = TaskQueue()


# ==================== ОПРЕДЕЛЕНИЕ УСТРОЙСТВА ====================
def get_optimal_device() -> tuple:
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"🚀 GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        if vram < 4:
            logger.warning("⚠️ Малый VRAM — включена агрессивная оптимизация")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        logger.info("🍎 Apple Silicon MPS найден")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("💻 GPU не найден — работаем на CPU")
    return device, dtype


# ==================== ЗАГРУЗКА МОДЕЛИ ====================
def load_pipeline(
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple:
    if model_id is None:
        model_id = Config.PRIMARY_MODEL_ID
    if device is None or dtype is None:
        device, dtype = get_optimal_device()

    logger.info(f"📥 Загрузка модели {model_id} → {device}...")
    t0 = time.time()

    def _try_load(mid: str) -> StableDiffusionImg2ImgPipeline:
        return StableDiffusionImg2ImgPipeline.from_pretrained(
            mid,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )

    pipe = None
    for attempt_id in [model_id] + Config.FALLBACK_MODELS:
        try:
            pipe = _try_load(attempt_id)
            model_id = attempt_id
            break
        except Exception as e:
            logger.warning(f"Не удалось загрузить {attempt_id}: {e}")

    if pipe is None:
        raise RuntimeError("Не удалось загрузить ни одну модель")

    if Config.ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()
        logger.info("✂️ Attention slicing включён")

    pipe = pipe.to(device)
    logger.info(f"🔄 Модель загружена на {device}")

    logger.info(f"✅ Модель загружена за {time.time() - t0:.1f} сек")
    return pipe, device, model_id


# ==================== ПРОГРЕВ МОДЕЛИ ====================
def warmup_pipeline(pipe: StableDiffusionImg2ImgPipeline, device: str) -> None:
    logger.info("🔥 Прогрев модели...")
    try:
        dummy = Image.new("RGB", (512, 512), color="black")
        ctx = torch.autocast(device_type=device) if device not in ("cpu", "mps") else nullcontext()
        with torch.inference_mode(), ctx:
            pipe(
                prompt="test",
                image=dummy,
                strength=0.3,
                num_inference_steps=2,
                guidance_scale=1.0,
            )
        logger.info("✅ Прогрев завершён")
    except Exception as e:
        logger.warning(f"⚠️ Прогрев не удался (не критично): {e}")


# ==================== ОБРАБОТКА ИЗОБРАЖЕНИЙ ====================
def prepare_image(image: Image.Image, size: int = Config.IMAGE_SIZE) -> Image.Image:
    """Конвертация в RGB + resize с letterbox"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(
        image,
        ((size - image.width) // 2, (size - image.height) // 2),
    )
    return canvas


def make_cache_key(data: bytes, style: str, steps: int, strength: float, guidance: float) -> str:
    raw = data + f"{style}{steps}{strength}{guidance}".encode()
    return hashlib.md5(raw).hexdigest()


# ==================== ГЕНЕРАЦИЯ ====================
async def _generate(
    state: dict,
    image: Image.Image,
    style: str,
    steps: int,
    strength: float,
    guidance: float,
) -> bytes:
    pipe = state["pipe"]
    device = state["device"]

    # На CPU автоматически снижаем шаги
    if device == "cpu" and steps > Config.CPU_FALLBACK_STEPS:
        logger.info(f"⚠️ CPU: снижаем steps {steps} → {Config.CPU_FALLBACK_STEPS}")
        steps = Config.CPU_FALLBACK_STEPS

    positive = STYLES[style]["positive"]
    negative = STYLES[style]["negative"]
    logger.info(f"🎨 Генерация: style={style}, steps={steps}, strength={strength}, guidance={guidance}")
    t0 = time.time()

    def _sync() -> bytes:
        # torch.autocast принимает device_type (не device объект)
        device_type = device if device not in ("mps",) else "cpu"
        ctx = torch.autocast(device_type=device_type) if device == "cuda" else nullcontext()
        with torch.inference_mode(), ctx:
            result = pipe(
                prompt=positive,
                negative_prompt=negative,
                image=image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    try:
        result_bytes = await asyncio.wait_for(
            run_in_threadpool(_sync),
            timeout=Config.GENERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise Exception(f"Генерация превысила таймаут {Config.GENERATION_TIMEOUT} сек")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise Exception("Недостаточно памяти GPU. Уменьшите steps или strength")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"✅ Готово за {time.time() - t0:.1f} сек")
    return result_bytes


# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    logger.info("=" * 55)
    logger.info("🚀 Запуск Blythe Doll Generator Server")
    logger.info("=" * 55)

    worker_task: Optional[asyncio.Task] = None

    try:
        pipe, device, model_id = load_pipeline()
        pipeline_state.update(
            pipe=pipe,
            device=device,
            model_id=model_id,
            ready=True,
        )

        # Прогрев в фоне
        asyncio.create_task(run_in_threadpool(warmup_pipeline, pipe, device))

        # Запуск воркера очереди
        worker_task = asyncio.create_task(task_queue.worker(pipeline_state))

        # Heartbeat для Render/HF
        if Config.HEARTBEAT_URL:
            asyncio.create_task(_heartbeat_loop())

        dev_label = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else device.upper()
        await send_telegram_message(f"✅ Сервер запущен\nУстройство: {dev_label}\nМодель: {model_id}")
        logger.info("✅ Сервер готов к работе")

    except Exception as e:
        logger.error(f"❌ Критическая ошибка при запуске: {e}")
        pipeline_state["ready"] = False
        await send_telegram_message(f"❌ Ошибка запуска: {str(e)[:200]}")

    yield  # Сервер работает

    # ---- Shutdown ----
    logger.info("🛑 Остановка сервера...")

    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    if pipeline_state["pipe"]:
        del pipeline_state["pipe"]
        pipeline_state["pipe"] = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache.clear()
    logger.info("👋 Сервер остановлен")
    await send_telegram_message("🛑 Сервер остановлен")


async def _heartbeat_loop() -> None:
    """Heartbeat чтобы Render/HF не усыплял сервис"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            await asyncio.sleep(Config.HEARTBEAT_INTERVAL)
            try:
                await client.get(Config.HEARTBEAT_URL)
                logger.debug("Heartbeat sent")
            except Exception:
                pass  # Некритично


# ==================== СОЗДАНИЕ ПРИЛОЖЕНИЯ ====================
app = FastAPI(
    title="Blythe Doll Generator",
    description="Преобразование фотографий в стиль куклы Blythe с помощью Stable Diffusion",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        f"https://{ALLOWED_DOMAIN}",
        f"https://www.{ALLOWED_DOMAIN}",
        "https://*.huggingface.co",
        "https://*.render.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - t0:.3f}"
    return response


# ==================== ЭНДПОИНТЫ ====================
@app.get("/")
async def root():
    return {
        "name": "Blythe Doll Generator API",
        "version": "2.1.0",
        "status": "operational" if pipeline_state["ready"] else "starting",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "styles": "GET /styles",
            "generate": "POST /generate",
            "queue_status": "GET /queue/{task_id}",
        },
    }


@app.get("/health")
async def health_check():
    device = pipeline_state["device"]
    gpu_info: dict = {}
    if device == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_info = {
            "gpu_memory_free_mb": int(free / 1024 / 1024),
            "gpu_memory_total_mb": int(total / 1024 / 1024),
        }

    proc = psutil.Process()
    return {
        "status": "ready" if pipeline_state["ready"] else "loading",
        "device": device,
        "model": pipeline_state.get("model_id", Config.PRIMARY_MODEL_ID),
        "styles": list(STYLES.keys()),
        "uptime_seconds": int(time.time() - pipeline_state["start_time"]),
        "queue_size": task_queue.queue.qsize(),
        "active_tasks": task_queue.active_count,
        "cache_size": len(cache),
        "ram_usage_mb": int(proc.memory_info().rss / 1024 / 1024),
        "cpu_percent": proc.cpu_percent(interval=0.1),
        **gpu_info,
    }


@app.get("/styles")
async def get_styles():
    return {
        name: {
            "name": name.capitalize(),
            "description": f"Преобразование в стиль {name}",
            "positive_prompt_preview": info["positive"][:100] + "...",
            "negative_prompt": info["negative"],
        }
        for name, info in STYLES.items()
    }


@app.get("/queue/{task_id}")
async def get_task_status(task_id: str):
    task = task_queue.get_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    resp: dict = {
        "task_id": task_id,
        "status": task.status,
        "created_at": task.created_at,
    }
    if task.started_at:
        resp["started_at"] = task.started_at
    if task.completed_at:
        resp["completed_at"] = task.completed_at
        resp["processing_seconds"] = round(task.completed_at - (task.started_at or task.created_at), 2)
    if task.error:
        resp["error"] = task.error
    if task.status == GenerationStatus.COMPLETED and task.result_bytes:
        resp["result_url"] = f"/result/{task_id}"
    return resp


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Получить готовый результат по task_id"""
    task = task_queue.get_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != GenerationStatus.COMPLETED or not task.result_bytes:
        raise HTTPException(status_code=404, detail="Result not ready yet")
    return Response(
        content=task.result_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=blythe_{task_id}.png"},
    )


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    style: str = Form("blythe"),
    steps: int = Form(Config.DEFAULT_STEPS),
    strength: float = Form(Config.DEFAULT_STRENGTH),
    guidance: float = Form(Config.DEFAULT_GUIDANCE),
):
    """Принять фото и вернуть стилизованное изображение (или task_id если занято)"""

    if not pipeline_state["ready"] or pipeline_state["pipe"] is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model loading", "message": "Модель загружается, попробуйте позже"},
        )

    if style not in STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный стиль: {style}. Доступны: {', '.join(STYLES.keys())}",
        )

    # Клиппинг параметров
    steps = max(Config.MIN_STEPS, min(Config.MAX_STEPS, steps))
    strength = max(Config.MIN_STRENGTH, min(Config.MAX_STRENGTH, strength))
    guidance = max(Config.MIN_GUIDANCE, min(Config.MAX_GUIDANCE, guidance))

    # Проверка размера
    if file.size and file.size > Config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"Файл слишком большой. Максимум {Config.MAX_FILE_SIZE_MB} МБ")

    # Чтение с таймаутом
    try:
        contents = await asyncio.wait_for(file.read(), timeout=Config.FILE_READ_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Таймаут чтения файла")

    if len(contents) > Config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"Файл слишком большой. Максимум {Config.MAX_FILE_SIZE_MB} МБ")

    # Валидация изображения
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Неверный формат файла. Ожидается JPEG или PNG")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения изображения: {e}")

    # Проверка кэша
    cache_key = make_cache_key(contents, style, steps, strength, guidance)
    cached = cache.get(cache_key)
    if cached:
        return Response(
            content=cached,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=blythe_doll.png",
                "X-Cache": "HIT",
            },
        )

    # Подготовка изображения
    processed = prepare_image(image)

    # Если сервер свободен — генерируем сразу
    if task_queue.active_count == 0 and task_queue.queue.empty():
        try:
            result_bytes = await _generate(pipeline_state, processed, style, steps, strength, guidance)
            cache.set(cache_key, result_bytes)
            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={
                    "Content-Disposition": "attachment; filename=blythe_doll.png",
                    "X-Cache": "MISS",
                },
            )
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}", exc_info=True)
            await send_telegram_message(f"❌ Ошибка генерации: {str(e)[:200]}")
            raise HTTPException(status_code=500, detail=str(e))

    # Иначе — ставим в очередь
    task_id = hashlib.md5(f"{cache_key}{time.time()}".encode()).hexdigest()[:16]
    position = await task_queue.enqueue(task_id, processed, {
        "style": style,
        "steps": steps,
        "strength": strength,
        "guidance": guidance,
    })

    return JSONResponse(
        status_code=202,
        content={
            "status": "queued",
            "task_id": task_id,
            "position": position,
            "message": f"Запрос в очереди. Позиция: {position}",
            "check_status": f"/queue/{task_id}",
            "get_result": f"/result/{task_id}",
        },
    )


# ==================== ЗАПУСК ====================
def _detect_env() -> tuple[str, int]:
    try:
        import google.colab  # noqa: F401
        return "colab", 7860
    except ImportError:
        pass
    if "SPACE_ID" in os.environ or "HF_SPACE" in os.environ:
        return "huggingface", int(os.environ.get("PORT", 7860))
    return "local", Config.PORT


def main():
    env, port = _detect_env()
    labels = {"colab": "Google Colab", "huggingface": "Hugging Face Spaces", "local": "Локальный"}
    logger.info(f"🌐 Среда: {labels[env]}")
    uvicorn.run(
        "doll_server:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Только 1 воркер для ML-моделей!
        log_level="info",
    )


if __name__ == "__main__":
    main()
