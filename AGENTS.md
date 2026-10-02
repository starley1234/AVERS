# AGENTS.md — инструкция для коддинг-агентов, работающих в этом репозитории

Этот файл — чтобы следующий агент (или вы сами на другой машине) не тратил
токены/время на то, что уже выяснено. Прочитайте его целиком перед тем, как
что-то менять.

## Что такое АВЕРС (1 абзац)

АВЕРС превращает растровые сканы схем бортовых кабельных сетей (БКС) в
граф связей (Netlist JSON/XML) через 6-стадийный пайплайн: SAHI-нарезка →
детекция УГО (RT-DETR/YOLO) → OCR (PaddleOCR, ГОСТ regex) → векторизация
линий (skeletonize+RDP) → графовый синтез (NetworkX) → VLM-арбитраж
коллизий (Vision RAG + few-shot). Есть Web UI валидатор, синтетический
генератор датасета по ГОСТ УГО, и (с этой сессии) полноценный v0.3 Active
Learning Loop. Подробности: `README.md`, `docs_v02.md`, `docs/ACTIVE_LEARNING_V03.md`.

## Текущее состояние (на момент этой правки)

- v0.1, v0.2 — готовы (production pipeline, Web UI, synthetic dataset, Vision RAG).
- v0.3 (Active Learning Loop) — **вся CPU-часть реализована и протестирована
  в этой сессии** (scheduler, model registry, notifications, dataset merge,
  dashboard, CLI). Единственное, что требует GPU — собственно запуск
  обучения на реальных данных. Подробный план и чеклист: **`docs/ACTIVE_LEARNING_V03.md`**.
- Каталог реальных (не синтетических) референсных схем для обучения:
  `data/reference_schematics/` + `docs/REAL_SCHEMATICS_SOURCES.md`.

**Перед тем как начинать новую задачу — прочитайте `docs/ACTIVE_LEARNING_V03.md`,
если она касается active learning/обучения моделей.** Там расписано, что уже
сделано, чтобы не дублировать работу.

## Как быстро поднять окружение (не тратьте токены на угадывание)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core deps, ~30 сек, работает без GPU/интернета к ML-хабам
python -m pytest tests/ -q                 # должно быть "140 passed" (на момент этой правки) без единой ML-библиотеки
```

Это баз для 95% задач (web UI, dataset tools, active learning логика, графы,
OCR regex, векторизация). **НЕ ставьте `ultralytics/torch/paddleocr/transformers/faiss`**,
если задача не требует реального обучения/инференса ML-моделей — пайплайн и
тесты построены с fallback-моками именно для этого (SAHI→mock, PaddleOCR→EasyOCR→mock,
CLIP→HOG, ultralytics отсутствует → понятная ошибка вместо падения). Установка
этих пакетов — это сотни МБ и минуты времени, которые не нужны для 95% правок.

Если задача всё же требует ML extras:
```bash
pip install ultralytics sahi paddleocr transformers torch faiss-cpu
```
(лучше на GPU-машине; `torch` под CUDA ставится отдельно под конкретную версию CUDA).

### ⚠️ Известная ловушка: opencv

`requirements.txt`/`pyproject.toml` используют **только** `opencv-python-headless`
(не `opencv-python`). Если по ошибке поставите оба — `pip uninstall` одного
сломает `cv2` для другого (они делят нативные библиотеки). Если видите
`ImportError: libGL.so.1: cannot open shared object file` — значит кто-то
поставил `opencv-python` в окружении без X11/libGL (это норма для
headless-песочниц/CI/серверов); решение: `pip uninstall opencv-python &&
pip install --force-reinstall --no-deps opencv-python-headless`.

### Известный факт: песочница без интернета к apt

В некоторых песочницах `apt-get install` не работает (нет доступа к
Debian-зеркалам). Если нужен системный пакет (libgl1 и т.п.) и apt не
работает — ищите чисто-python/headless альтернативу вместо системной
библиотеки (см. opencv выше).

## Структура репозитория (коротко, см. README.md для полной версии)

```
avers/
  core/            # Pydantic-типы, ProductionPipeline, валидаторы
  stages/          # 6 стадий пайплайна
  web/             # FastAPI: api.py (основное+active-learning stats),
                   #   active_learning_api.py (scheduler/registry/dashboard - NEW v0.3),
                   #   annotator_api.py, app.py, frontend/*.html
  dataset/         # synthetic.py, generator.py, gost_symbols.py, train.py,
                   #   public_datasets.py, merge.py (NEW v0.3), real_schematics.py (NEW v0.3)
  active_learning/ # loop.py, metrics.py, registry.py (NEW), scheduler.py (NEW), notify.py (NEW)
  rag/             # CLIP+FAISS Vision RAG
tests/             # pytest, зеркалит структуру avers/ 1:1 по файлам
docs/              # ACTIVE_LEARNING_V03.md (NEW), REAL_SCHEMATICS_SOURCES.md (NEW)
data/reference_schematics/  # NEW: куратированные примеры реальных схем RU/СССР
config.yaml        # полный конфиг со всеми секциями, включая active_learning (NEW)
```

## Правила, чтобы не расходовать токены впустую

1. **Сначала читайте, потом ищите.** `README.md`, `ROADMAP.md`,
   `docs/ACTIVE_LEARNING_V03.md`, `PUBLIC_DATASETS.md`,
   `docs/REAL_SCHEMATICS_SOURCES.md` уже содержат ответы на большинство
   вопросов "а что тут вообще происходит" и "что уже пробовали".
2. **Запускайте целевые тесты, а не весь набор**, если правите один модуль:
   `python -m pytest tests/test_active_learning.py -q` (~1 сек) вместо
   `tests/` целиком (~90 сек, т.к. test_full_pipeline/test_production гоняют
   настоящие (хоть и маленькие) изображения через весь pipeline).
3. **Не переустанавливайте venv/зависимости каждый раз** — если `.venv/`
   уже есть и `pip show <pkg>` подтверждает нужную версию, просто
   активируйте: `source .venv/bin/activate`.
4. **Не коммитьте веса моделей, большие датасеты, `/tmp/avers_*`.**
   `.gitignore` уже исключает `*.pt/*.onnx`, venv, кеши. Файлы
   `data/reference_schematics/` — осознанное исключение (маленький
   курированный набор, см. его README про лицензии) — не плодите там
   большие "сырые" дампы, кладите их в `data/reference_schematics/<tier>/raw/`
   (уже в `.gitignore`).
5. **Active learning / обучение моделей = GPU-машина.** Эта песочница (и,
   возможно, та, в которой вы сейчас читаете этот файл) обычно без GPU —
   код пишется и тестируется здесь (все модули v0.3 спроектированы
   CPU-testable), а реальные `avers dataset train` / `avers active-learning
   retrain` без `--dry-run` гоняются на машине с CUDA.
6. **Если меняете `avers/web/api.py` или добавляете роуты** — помните, что
   `/api/active-learning/*` теперь разделён между `avers/web/api.py`
   (старые: stats/retrain/clear, из v0.2) и `avers/web/active_learning_api.py`
   (новые: scheduler/registry/history/notifications, v0.3). Не дублируйте
   пути между этими двумя роутерами.
7. **Коммитьте и пушьте в ветку `arena/01a0fbb1-avers`** (эта сессия
   привязана к ней) — не создавайте других веток.

## Как проверить, что ничего не сломали, перед тем как закончить

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # 140 passed, 0 failed (без ML extras)
python -m avers active-learning stats                       # CLI работает
python -c "from avers.web.app import create_app; create_app()"  # web app собирается
```
