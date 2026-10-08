# AGENTS.md — инструкция для коддинг-агентов, работающих в этом репозитории

**НОВАЯ СЕССИЯ: сначала прочитайте docs/HANDOFF.md** (состояние прототипа, что не протестировано, план по приоритетам). Windows: start.cmd/stop.cmd, окружение env\\, тесты env\\Scripts\\python.exe -m pytest tests -q (248 passed).


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
- **v0.4 (эта правка): работа без обученных весов.** Библиотека УГО расширена
  до 31 класса реальных элементов по ЕСКД. Резистор теперь по ГОСТ 2.728
  (прямоугольник), а не ANSI-зигзаг. Добавлен CPU-детектор по шаблонам
  библиотеки, демо-схема БКС с эталонным netlist (`data/demo_bks_schematic.*`)
  и кнопка «Демо-схема» в Web UI. Демо даёт 19/19 компонентов и 11/11 цепей.
  Всё описано в **`docs/GOST_LIBRARY_AND_TEMPLATE_DETECTOR.md`**.
  С установленным EasyOCR (`pip install easyocr`) позиционные обозначения
  (R1, VD1, K1.1) берутся со схемы (`avers/core/designators.py`), и на демо
  читаются все 14 из 14. Без EasyOCR тест с OCR пропускается; подставной тест
  OCR работает всегда. Windows: `start.cmd` / `stop.cmd`.
  Обученные веса подключаются через ultralytics с нарезкой на тайлы и
  гибридным уточнением выводов по шаблонам ГОСТ (`model_detector.py`, раздел в
  том же doc). SAHI больше не используется. **ultralytics ставить только
  `--no-deps`**, иначе он притянет `opencv-python` (см. ловушку opencv ниже).
  Новые классы
  добавляйте в `GOST_SYMBOLS` и зеркально в `avers/core/types.py::DETECTION_CLASSES`
  (это проверяет тест).
- Каталог реальных (не синтетических) референсных схем для обучения:
  `data/reference_schematics/` + `docs/REAL_SCHEMATICS_SOURCES.md`.

**Перед тем как начинать новую задачу — прочитайте `docs/ACTIVE_LEARNING_V03.md`,
если она касается active learning/обучения моделей.** Там расписано, что уже
сделано, чтобы не дублировать работу.

## Как быстро поднять окружение (не тратьте токены на угадывание)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core deps, ~30 сек, работает без GPU/интернета к ML-хабам
python -m pytest tests/ -q                 # должно быть "~248 passed" (на момент этой правки; тест с EasyOCR пропускается, если он не установлен) без единой ML-библиотеки
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
python -m pytest tests/ -q          # ~248 passed, 0 failed (без ML extras), ~60-100 сек
ruff check avers/ --select E9,F821,F823,F811  # реальные баги (не стиль) — должно быть "All checks passed!"
python -m avers active-learning stats                       # CLI работает
python -c "from avers.web.app import create_app; create_app()"  # web app собирается
```

Есть CI (`.github/workflows/tests.yml`): на каждый push/PR гоняет pytest +
точечный ruff-линт (`E9,F821,F823,F811` — синтаксис и реально неопределённые
имена, НЕ стилевые правила) на Python 3.11 и 3.12, без единой ML-зависимости.

## Известные/уже исправленные ловушки (чтобы не наступать повторно)

- `ruff check avers/ --select F821,F823,F811` один раз уже нашёл реальный баг:
  в `avers/stages/stage5_graph_synthesis/graph_builder.py` метод
  `GraphBuilder.visualize()` падал с `UnboundLocalError: cv2` при любом
  вызове — локальный `import cv2` внутри `if output_path:` делал имя `cv2`
  function-local для всего метода целиком (включая более ранние вызовы
  `cv2.line/circle`). Метод не вызывался нигде в коде/тестах, поэтому 140
  прошедших тестов этого не ловили. Исправлено: `import cv2` перенесён на
  уровень модуля; добавлен регрессионный тест
  `tests/test_core.py::TestGraphBuilder::test_visualize_returns_image`.
  **Вывод:** весь `avers/` никогда не прогонялся через `ruff`/`mypy` до этой
  правки — весь остальной `--statistics` вывод (~1000 находок) это в основном
  стилевые/модернизационные вещи (non-pep585 аннотации, неотсортированные
  импорты), НЕ баги; не тратьте токены на их массовое исправление одним
  большим диффом без запроса пользователя — риск конфликтов выше пользы.
- `avers/dataset/synthetic.py` содержал мёртвый no-op
  `GOSTGenerator = GOSTGenerator` (ничего не алиасил) — удалён.
- Голые `except:` в `avers/rag/store.py` и `avers/web/annotator_api.py`
  заменены на `except Exception:` (чтобы не глотать `KeyboardInterrupt`/
  `SystemExit`). Остальные `except Exception: pass` в `avers/web/api.py` и
  др. — осознанный fallback-паттерн (graceful degradation при отсутствии
  RAG/FAISS), трогать не нужно.
