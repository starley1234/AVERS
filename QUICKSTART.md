# AVERS — Быстрый старт

> Подробности, архитектура и все возможности — в [README.md](README.md).
> Эта инструкция — только про то, как быстро запустить и начать пользоваться.

---

## 0. Требования

- Linux / WSL2 / macOS, Python **3.11+** (`python3 --version`)
- 4+ ГБ RAM для пайплайна; GPU (CUDA) — только для быстрого обучения, всё остальное работает на CPU
- Интернет для первого запуска (скачивание пакетов и весов моделей)

---

## 1. Один скрипт — и готово (рекомендуется)

```bash
cd AVERS
bash scripts/quickstart.sh
```

Скрипт сам:
1. создаст/обновит виртуальное окружение `.venv`,
2. установит зависимости и проверит целостность OpenCV,
3. сгенерирует демонстрационную схему ГОСТ,
4. прогонит через неё полный пайплайн (6 стадий) и сохранит результат,
5. напечатает, что делать дальше.

Полезные флаги:

```bash
bash scripts/quickstart.sh --web       # в конце запустить Web UI (http://localhost:8000)
bash scripts/quickstart.sh --with-ml   # + ML-зависимости (ultralytics, sahi, transformers)
bash scripts/quickstart.sh --full      # полный синтетический датасет (1000/200/100, дольше)
bash scripts/quickstart.sh --skip-install   # не трогать окружение (уже настроено)
```

---

## 2. Если вручную

```bash
cd AVERS
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# ML-зависимости (нужны для обучения и полной стадии детекции):
pip install ultralytics sahi

# Проверка, что всё живо:
bash scripts/smoke_test.sh
```

### Сценарий А — обработать схему (распознавание)

```bash
# Своё изображение/PDF (TIF, PNG, JPG, PDF):
python -m avers process мой_скан.tif -o result.json
python -m avers process мой_скан.pdf --pdf-page 1 -o result.json

# Или на демонстрационной схеме, созданной quickstart'ом:
python -m avers process /tmp/avers_demo/previews/preview_0000.jpg -o /tmp/avers_demo/result.json
```

Результат — netlist JSON (компоненты, связи, тексты). XML: добавьте `-f xml`.
Просмотр результата в браузере: `python -m avers web` → вкладка «Визуализация».

### Сценарий Б — Web UI валидатор

```bash
python -m avers web --host 0.0.0.0 --port 8000
# открыть http://localhost:8000
```

Drag&drop схемы → пайплайн → просмотр графа, правка, экспорт.

### Сценарий В — синтетический датасет + обучение детектора

```bash
# 1. Датасет (быстрый вариант для проверки: 50/10/5 картинок)
python -m avers dataset generate -o /tmp/avers_dataset \
    --num-train 50 --num-val 10 --num-test 5 --image-size 640

# 2. Быстрая проверка обучения (1 эпоха, лёгкая модель — минуты даже на CPU)
python -m avers dataset train --data /tmp/avers_dataset/dataset.yaml \
    --model yolo11n --epochs 1 --batch 4

# 3. Боевое обучение (GPU)
python -m avers dataset train --data /tmp/avers_dataset/dataset.yaml \
    --model rtdetr-l --epochs 100 --batch 8
```

- `--device` по умолчанию `auto`: возьмёт CUDA, если есть, иначе CPU (раньше был жёсткий `cuda` и падал без GPU).
- Результаты: `/tmp/avers_runs/avers_yolo|avers_rtdetr/weights/best.pt` + `best.onnx`.
- Чтобы пайплайн использовал обученную модель — пропишите путь в `config.yaml`:
  `detection.model_path: /tmp/avers_runs/avers_yolo/weights/best.pt`.

---

## 3. Внешняя LLM (VLM-арбитраж, стадия 6)

Поддерживается любой **OpenAI-совместимый API с поддержкой картинок**: OpenAI, OpenRouter,
vLLM, Ollama (`/v1`), LM Studio и т.п. Задаётся переменными окружения:

| Переменная | Смысл | Пример |
|---|---|---|
| `AVERS_VLM_API_BASE` | базовый URL API | `https://api.openai.com/v1` |
| `AVERS_VLM_API_KEY` | ключ (Bearer) | `sk-...` |
| `AVERS_VLM_API_MODEL` | имя модели с vision | `gpt-4o-mini` |
| `AVERS_VLM_PROVIDER` | `auto`/`api`/`local`/`mock` | `auto` (API, если задан base) |
| `AVERS_VLM_API_TIMEOUT` | таймаут, сек | `60` |

Примеры:

```bash
# OpenAI
export AVERS_VLM_API_BASE=https://api.openai.com/v1
export AVERS_VLM_API_KEY=sk-...
export AVERS_VLM_API_MODEL=gpt-4o-mini

# OpenRouter
export AVERS_VLM_API_BASE=https://openrouter.ai/api/v1
export AVERS_VLM_API_KEY=sk-or-...
export AVERS_VLM_API_MODEL=qwen/qwen2.5-vl-72b-instruct

# Локальная Ollama
export AVERS_VLM_API_BASE=http://localhost:11434/v1
export AVERS_VLM_API_MODEL=qwen2.5vl:7b
```

Можно и без env — в `config.yaml`, секция `vlm_arbitrator:` (`api_base`, `api_key`, `api_model`).
Если внешний API не задан, используется локальная модель из `model_name` (нужен `transformers`
и GPU), а если и её нет — детерминированная заглушка (mock), пайплайн всё равно работает.

---

## 4. Docker

```bash
docker build -t avers .
docker run -p 8000:8000 avers                     # Web UI на http://localhost:8000
docker run --rm -v $(pwd):/data avers process /data/scan.tif -o /data/result.json
docker compose up -d                              # Web UI + (опц.) qdrant/minio: --profile with-qdrant
```

Собран ли образ на вашей машине: `docker images | grep avers`. Если пусто — не собран, собирайте
командой выше. GPU-версия требует `nvidia-container-toolkit` и флаг `--gpus all`.

---

## 5. Проверка работоспособности + лог для анализа

```bash
bash scripts/smoke_test.sh
```

Скрипт прогоняет все подсистемы (окружение, генерация датасета, полный пайплайн, RAG,
микро-обучение 1 эпохи, Web API, внешний LLM если настроен), складывает всё в один файл
`avers_smoke_ГГГГММДД_ЧЧММСС.log` рядом с репозиторием и печатает итоговую таблицу.
Пришлите этот файл для анализа. Флаги: `--skip-train`, `--skip-web`, `--with-rtdetr`.

---

## 6. Частые проблемы

| Симптом | Причина / решение |
|---|---|
| `ImportError: libGL.so.1` | Установился `opencv-python` вместо headless: `pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless` |
| `device='cuda' but CUDA unavailable` | Нет GPU → используйте `--device auto` или `--device cpu` (теперь это дефолт) |
| `ultralytics ... got multiple values for keyword argument 'epochs'` | Исправлено в `avers/dataset/train.py` — обновите код |
| Обучение на Python 3.13 не ставит paddleocr | PaddleOCR официально не поддерживает 3.13 — OCR-стадия перейдёт в заглушку, остальное работает; лучше venv на 3.11/3.12 |
| Медленно качаются веса | Веса моделей (`rtdetr-l.pt` 63 МБ) качаются с GitHub один раз; можно положить `.pt` рядом с местом запуска вручную |
