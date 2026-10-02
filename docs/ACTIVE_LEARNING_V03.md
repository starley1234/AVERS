# v0.3 — Active Learning Loop: статус и план работ для GPU-машины

> Этот документ — **точка входа для агента/инженера, который продолжит
> работу над v0.3 на машине с GPU**. Он написан так, чтобы не пришлось заново
> изучать репозиторий: что уже готово и протестировано (можно использовать
> как есть), что нужно доделать, и в каком порядке. См. также `ROADMAP.md`
> (общая дорожная карта) и `AGENTS.md` (как вообще работать в этом репо).

## Коротко: что такое Active Learning Loop

```
Web UI валидатор → пользователь исправляет ошибку (FeedbackEntry)
   → сохраняется в /tmp/avers_feedback/ + индексируется в Vision RAG
   → накапливается N исправлений → "retrain due"
   → merge(synthetic GOST + накопленный feedback [+ public]) → dataset.yaml
   → fine-tune RT-DETR/YOLO (мало эпох, т.к. это дообучение, не обучение с нуля)
   → новая версия модели регистрируется в Model Registry
   → promote в прод (или A/B тест против текущей) → ProductionPipeline подхватывает новые веса
   → цикл повторяется
```

## Что уже готово (CPU-only, протестировано, можно использовать сразу)

Всё из списка ниже было реализовано/проверено в этой сессии **без GPU** —
логика полностью CPU-only и не зависит от torch/ultralytics (они опциональны
и подключаются лениво только в момент реального обучения). 140/140 тестов
проходят (`python -m pytest tests/ -q`).

| Компонент | Файл | Статус |
|---|---|---|
| Сбор feedback от валидатора | `avers/active_learning/loop.py` (`ActiveLearningLoop`, `FeedbackEntry`) | ✅ было в v0.2, протестировано заново |
| Auto-index в Vision RAG при feedback | `loop.py` + `avers/rag/` | ✅ **починен баг**: `get_rag` не экспортировался из `avers/rag/__init__.py`, из-за чего RAG-индексация тихо падала. Исправлено. |
| Метрики feedback (per day, top corrections) | `avers/active_learning/metrics.py` | ✅ |
| **Model Registry** (версионирование, promote/rollback, A/B роутинг) | `avers/active_learning/registry.py` | ✅ новое, 9 unit-тестов |
| **Notifier** (retrain_due/started/done/failed, webhook + JSONL лог) | `avers/active_learning/notify.py` | ✅ новое, 3 unit-теста |
| **RetrainScheduler** (фоновый поток ИЛИ blocking cron-режим) | `avers/active_learning/scheduler.py` | ✅ новое, 5 unit-тестов |
| **Merge датасетов** (synthetic + feedback [+ public], с ремапом классов и oversampling) | `avers/dataset/merge.py` | ✅ новое, 4 unit-теста |
| Web API: scheduler start/stop/status, registry list/promote/rollback/ab-test, history, notifications | `avers/web/active_learning_api.py` | ✅ новое, проверено через TestClient |
| Простой dashboard (feedback, scheduler status, версии моделей) | `avers/web/frontend/dashboard.html` (`/dashboard`) | ✅ новое |
| CLI: `avers active-learning {stats,retrain,merge-datasets,scheduler,registry}` | `avers/main.py` | ✅ новое, вручную проверено (merge + export работают end-to-end на CPU) |
| Конфигурация (`active_learning:` секция) | `avers/config.py`, `config.yaml` | ✅ новое |
| Реальные референсные схемы (для затравки/smoke-теста) | `data/reference_schematics/` + `docs/REAL_SCHEMATICS_SOURCES.md` | ✅ новое, 12 изображений + каталог источников для масштабирования |

**Проверено вручную end-to-end на CPU** (без GPU, т.к. ultralytics не ставился
в этой песочнице — установка тяжёлая и не нужна для подготовительной работы):

```bash
python -m avers dataset generate --output /tmp/gost --num-train 4 --num-val 2
# ... добавили 2 FeedbackEntry через ActiveLearningLoop ...
python -m avers active-learning retrain --synthetic /tmp/gost/dataset.yaml --model yolo11x --epochs 1
# -> экспортировал feedback, смержил с synthetic (10 train / 8 val, 10 классов),
#    попытался обучить -> корректно сообщил "ultralytics not installed" и
#    вернул status=success для шага merge (само обучение - следующий шаг на GPU)
```

## Что осталось сделать на GPU-машине

Порядок важен: сначала дешёвые/быстрые шаги, потом дорогие (реальное обучение).

### 1. Окружение (5 минут)
```bash
git clone <repo> && cd AVERS
git checkout arena/01a0fbb1-avers   # эта ветка
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics sahi paddleocr transformers torch faiss-cpu  # [ml] extras, нужен CUDA-билд torch под вашу GPU
python -m pytest tests/ -q   # должно быть 140 passed (без GPU) - подтверждает что ничего не сломалось при установке ML-зависимостей
```
⚠️ Не ставьте одновременно `opencv-python` и `opencv-python-headless` (см.
комментарий в `requirements.txt`) — ломает `cv2` целиком.

### 2. Базовая синтетика + первая модель (несколько часов GPU)
```bash
python -m avers dataset generate --output /tmp/avers_gost --num-train 3000 --num-val 300
python -m avers dataset train --data /tmp/avers_gost/dataset.yaml --model rtdetr-l --epochs 100 --device cuda
```
Зарегистрируйте результат как `v000001` в Model Registry:
```bash
python -c "
from avers.active_learning.registry import get_model_registry
reg = get_model_registry()
v = reg.register('/tmp/avers_runs/avers_rtdetr/weights/best.pt', 'rtdetr-l', metrics={'mAP50': 0.0})  # впишите реальный mAP из логов обучения
print(v.version_id)
"
```
Пропишите путь в `config.yaml` (`detection.model_path`) или держите его в
Model Registry и подтягивайте через `get_model_registry().get_current().weights_path`
(рекомендуется — так ProductionPipeline не захардкожен на один путь).

### 3. Реальные данные: от простого к сложному
Следуйте `docs/REAL_SCHEMATICS_SOURCES.md`:
1. Разметьте уже committed `data/reference_schematics/tier1_simple_car/*`
   через `/annotator` (3 изображения - это только smoke-test, не датасет).
2. Скачайте больше схем Tier 1 (простые авто) по каталогу источников,
   разметьте 30-50 штук.
3. Смержите с synthetic (`avers/dataset/merge.py`, см. пример в roadmap-доке),
   дообучите (`python -m avers dataset train ... --pretrained <v000001 weights>`
   с меньшим `--epochs`).
4. Повторите для Tier 2 (радиосхемы), затем Tier 3 (грузовики), затем —
   **настоящие БКС-сканы заказчика**, если/когда они появятся.

### 4. Включить полный цикл в Web UI
```bash
export AVERS_AL_SCHEDULER_ENABLED=true   # запускает RetrainScheduler при старте приложения
export AVERS_NOTIFY_WEBHOOK_URL=https://hooks.slack.com/...  # опционально
python -m avers web --port 8000
# http://localhost:8000/dashboard  <- мониторинг feedback/scheduler/registry
```
Исправляйте проблемы в валидаторе → `resolve_issue` уже автоматически
добавляет feedback в `ActiveLearningLoop` и индексирует в RAG (см.
`avers/web/api.py::resolve_issue`). Когда накопится
`active_learning.min_feedback_for_retrain` исправлений (по умолчанию 50,
настраивается в `config.yaml`), шедулер (если включён) автоматически:
merge-ит feedback+synthetic → дообучает → регистрирует новую версию →
шлёт уведомление.

### 5. Оставшиеся инженерные доработки (не блокируют запуск, но стоит сделать)

- **Подключить `ProductionPipeline` к Model Registry**, чтобы она брала
  `detection.model_path` из `get_model_registry().get_current()` вместо
  статичного `config.yaml` пути — тогда promote/rollback реально переключают
  модель в проде без рестарта. Сейчас registry и pipeline существуют
  независимо (намеренно, чтобы не трогать production-критичный код без
  GPU-тестирования); связка — следующий шаг. Смотрите
  `avers/core/validators.py` / `avers/pipeline.py` (`ProductionPipeline`,
  детектор загружается из `config.detection.model_path`).
- **A/B тест в проде**: `ModelRegistry.route(request_id)` уже умеет детерминированно
  распределять трафик между `current` и `challenger` — нужно вызвать его в
  `avers/web/api.py` в месте, где выбирается модель для обработки конкретного
  файла, и логировать исход через `registry.log_ab_result(...)` (например, по
  тому, принял ли пользователь предсказание без правок в валидаторе).
- **mAP на валидации вместо placeholder метрик**: `train_rtdetr`/`train_yolo`
  (`avers/dataset/train.py`) уже вызывают `model.train(...)`; добавьте парсинг
  `results` (ultralytics возвращает объект с метриками) и прокиньте
  `metrics={"mAP50": ...}` в `ActiveLearningLoop.trigger_retraining()` →
  `RetrainScheduler._maybe_register()`, чтобы Model Registry хранил реальные
  числа (сейчас возвращается пустой dict `{}` — заглушка, явно размечено TODO
  по месту, где это нужно донастроить под ваш формат вывода ultralytics).
- **Публичные датасеты как ещё один источник для `merge_datasets`**: сейчас
  `avers/dataset/public_datasets.py::create_mixed_dataset_config` и
  `avers/dataset/merge.py::merge_datasets` решают очень похожую задачу двумя
  разными путями (исторически). Стоит унифицировать — например, сделать
  `create_mixed_dataset_config` тонкой обёрткой над `merge_datasets` с
  `DatasetSource`. Не критично, но уменьшит дублирование.
- **Персистентность реестра/feedback**: сейчас `/tmp/avers_*` — это ОК для
  разработки, но на реальном сервере стоит вынести `feedback_dir`,
  `registry_dir`, `rag.storage_path` на постоянный диск (см. `config.yaml`,
  уже вынесено в конфиг — просто поменяйте пути с `/tmp/...` на постоянные).

## Контрольный список приёмки v0.3 (roadmap)

Из `ROADMAP.md`:
- [x] ActiveLearningLoop base
- [x] Web UI integration for feedback
- [x] Auto retraining scheduler (cron/background job) — `RetrainScheduler`
- [x] Model versioning and A/B testing — `ModelRegistry`
- [x] Metrics dashboard — `/dashboard` + `/api/active-learning/history`
- [x] Notification when retrain needed — `Notifier`
- [x] Merge synthetic + feedback datasets for training — `avers/dataset/merge.py`
- [ ] **Обучить и провалидировать реальную модель на GPU** (единственный
      оставшийся пункт, требующий GPU — всё остальное реализовано и
      протестировано на CPU в этой сессии)
- [ ] Подключить Model Registry к ProductionPipeline (см. раздел 5 выше)
- [ ] Реальные метрики mAP вместо заглушки в registry

## Как тестировать без GPU (уже сделано, для справки)

```bash
python -m pytest tests/test_active_learning.py -v   # 29 тестов: loop, metrics, registry, notify, scheduler, merge
python -m pytest tests/ -q                           # полный набор, 140 тестов
```
Все тесты в `tests/test_active_learning.py` используют `tmp_path` и
dependency-injected `retrain_fn`/`copy_weights=False` там, где нужно обойти
реальное обучение/torch — так что они будут проходить одинаково что на CPU,
что на GPU-машине, и являются живой спецификацией поведения модулей.
