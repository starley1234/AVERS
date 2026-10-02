# AVERS Roadmap

## ✅ v0.1 - Production Pipeline (Done)
- SAHI slicing 1024x1024 overlap 0.2
- RT-DETR/YOLO detection with fallback mock
- PaddleOCR with GOST regex
- Wire vectorization (skeletonize + RDP)
- Graph synthesis (NetworkX, snapping, nets)
- VLM arbitrator (Qwen-VL/Gemma with mock fallback)
- JSON/XML export
- ProductionPipeline with error handling, timings, validation
- 22 tests

## ✅ v0.2 - Web UI + Dataset + Vision RAG (Done - Current)
- **Web UI Validator**: FastAPI + vanilla JS, dark theme, drag&drop, zoom/pan canvas, SVG overlay, 5 stages viz, editing, export
- **Synthetic Dataset**: 10 GOST classes, GOSTGenerator, SchematicComposer A2x6, augmentations (scan noise, blur, vignette), YOLO/COCO export
- **Manual Annotator**: /annotator, bbox drawing, hotkeys 1-9/Del/Ctrl+S, train/val split
- **Training**: YOLOv11x/m, RT-DETRv2-l/x, GOST augmentations, ONNX export, CLI
- **Vision RAG**: CLIP ViT-B/32 512d + HOG fallback, FAISS + brute-force, few-shot VLM prompting, integration in Stage 6
- **Active Learning**: FeedbackEntry, ActiveLearningLoop with RAG auto-index, retrain trigger
- **Docker**: Dockerfile + docker-compose with qdrant/minio profiles
- **Tests**: 38 new tests (dataset, rag, web), quickstart.py, CI workflow

## 🔄 v0.3 - Active Learning Loop (CPU part done - GPU training remaining)

> Полный статус, что проверено, и пошаговый план для GPU-машины:
> **`docs/ACTIVE_LEARNING_V03.md`**. Реальные (не синтетические) схемы для
> обучения: **`docs/REAL_SCHEMATICS_SOURCES.md`** + `data/reference_schematics/`.

- [x] ActiveLearningLoop base (done, v0.2)
- [x] Web UI integration for feedback (done, v0.2 - resolve_issue adds to loop + RAG)
- [x] Auto retraining scheduler (cron/background job) — `avers/active_learning/scheduler.py` (`RetrainScheduler`), CLI `avers active-learning scheduler run`, API `/api/active-learning/scheduler/{start,stop,status}`
- [x] Model versioning and A/B testing — `avers/active_learning/registry.py` (`ModelRegistry`: register/promote/rollback/compare/start_ab_test/route), CLI `avers active-learning registry`, API `/api/active-learning/registry/*`
- [x] Metrics dashboard (accuracy improvement over time) — `/dashboard` (`avers/web/frontend/dashboard.html`) + `GET /api/active-learning/history`
- [x] Notification when retrain needed — `avers/active_learning/notify.py` (`Notifier`: log/webhook/JSONL), wired into scheduler
- [x] Merge synthetic + feedback datasets for training — `avers/dataset/merge.py` (`merge_datasets`, handles class remap + oversampling), CLI `avers active-learning merge-datasets` / `retrain --synthetic ...`
- [x] Bugfix: `get_rag` wasn't exported from `avers/rag/__init__.py`, silently breaking RAG auto-indexing on feedback — fixed
- [x] 29 new unit tests (`tests/test_active_learning.py`), all CPU-only (no torch/ultralytics/GPU required)
- [ ] **Train & validate a real model on a GPU machine** (the only remaining item that actually needs a GPU — everything above is implemented and tested without one)
- [ ] Wire `ModelRegistry.get_current()` into `ProductionPipeline`'s detection model loading (currently independent - see docs/ACTIVE_LEARNING_V03.md §5)
- [ ] Replace placeholder `metrics={}` on retrain with real mAP parsed from ultralytics training results

**CLI (v0.2 + v0.3):**
```bash
python -m avers web --port 8000  # validator auto-collects feedback; http://localhost:8000/dashboard
curl http://localhost:8000/api/active-learning/stats
curl -X POST http://localhost:8000/api/active-learning/retrain?model_type=rtdetr-l&epochs=20

# v0.3 additions
python -m avers active-learning stats
python -m avers active-learning merge-datasets --synthetic /tmp/gost/dataset.yaml --feedback /tmp/fb/dataset_feedback.yaml -o /tmp/merged
python -m avers active-learning retrain --synthetic /tmp/gost/dataset.yaml --model rtdetr-l --epochs 20 --register
python -m avers active-learning scheduler run --interval 3600          # blocking, cron/systemd-friendly
python -m avers active-learning registry list
python -m avers dataset real-schematics list --tier tier1_simple_car   # real RU/USSR schematic sources catalog
```

## 📋 v0.4 - Native CAD Export
- [ ] Max-SAPR native XML format (detailed schema research needed)
- [ ] KOMPAS-Electric native format
- [ ] Altium, KiCad netlist export
- [ ] Wire type/color mapping (БПВЛ, МГТФ)
- [ ] Component library mapping (СНЦ, 2РМ, etc.)

## 📋 v0.5 - Multi-page & Large Format
- [ ] A2x6 stitching (multiple scans → single schematic)
- [ ] Offpage connector resolution (стрелки перехода)
- [ ] Multi-page net continuity
- [ ] PDF multi-page support
- [ ] Large image streaming (don't load full 15000x4000 into RAM)

## 📋 v0.6 - Production RAG
- [ ] Qdrant / Milvus integration (replace FAISS for prod)
- [ ] CLIP fine-tuning on GOST symbols
- [ ] Text embeddings for wire labels (БПВЛ, МГТФ)
- [ ] Hybrid search (vector + keyword)
- [ ] RAG evaluation metrics

## 📋 v0.7 - Advanced Detection
- [ ] SAM (Segment Anything) for auto-assisted annotation
- [ ] Table detection for connector pinouts
- [ ] Text-to-pin association with transformers (LayoutLM)
- [ ] Wire crossing classification with GNN
- [ ] End-to-end trainable pipeline

## 📋 v0.8 - Scale & Performance
- [ ] Distributed training (multi-GPU, multi-node)
- [ ] ONNX Runtime optimization
- [ ] TensorRT export
- [ ] Caching layer for embeddings
- [ ] Horizontal scaling for Web UI (k8s)

## 📋 v0.9 - Enterprise
- [ ] User auth & RBAC
- [ ] Project management (multiple schematics)
- [ ] Audit log
- [ ] S3/MinIO storage backend
- [ ] PostgreSQL for metadata

## 📋 v1.0 - Release
- [ ] Full documentation (user guide, API docs, training guide)
- [ ] Benchmarks on real BKS schematics
- [ ] Performance optimization
- [ ] Security audit
- [ ] PyPI release

---

## Что нужно сделать сейчас? (Приоритеты)

### Высокий приоритет (для v0.3 / MVP) — см. docs/ACTIVE_LEARNING_V03.md
1. **Обучить базовую модель на GPU** - `avers dataset generate` (крупный, 3000+) + `avers dataset train` на реальном GPU, зарегистрировать в Model Registry
2. **Собрать реальный датасет БКС-аналогов** - начать с `docs/REAL_SCHEMATICS_SOURCES.md` (Tier 1: простые авто-схемы → Tier 2: радиосхемы → Tier 3: грузовики), разметить через /annotator; `data/reference_schematics/` уже содержит 12 стартовых примеров
3. **Запустить Active Learning end-to-end на реальных данных** - `avers web` с `AVERS_AL_SCHEDULER_ENABLED=true`, собрать feedback, дать scheduler'у дообучить и зарегистрировать новую версию
4. **Подключить Model Registry к ProductionPipeline** - чтобы promote/rollback реально переключали модель в проде (сейчас независимые компоненты, см. docs/ACTIVE_LEARNING_V03.md §5)
5. **Протестировать на реальных сканах заказчика** (если доступны) - прогнать через pipeline, собрать feedback - это золотой стандарт, сильнее любого открытого аналога

### Средний приоритет
6. **Docker production** - протестировать docker-compose с GPU
7. **Экспорт в САПР** - исследовать форматы Max-САПР, КОМПАС
8. **Улучшить OCR** - дообучить PaddleOCR на ГОСТ шрифтах (ГОСТ 2.304)
9. **Добавить больше ГОСТ символов** - реле, контакторы, etc.
10. **Реальные mAP метрики в Model Registry** вместо заглушки (см. docs/ACTIVE_LEARNING_V03.md)

### Низкий приоритет (nice to have)
11. **SAM integration** для аннотатора
12. **Qdrant** вместо FAISS
13. **K8s deployment**
14. Унифицировать `avers/dataset/public_datasets.py::create_mixed_dataset_config` и `avers/dataset/merge.py::merge_datasets` (сейчас решают похожую задачу двумя путями по историческим причинам)

---

## Как помочь?

- Разметьте схемы через http://localhost:8000/annotator
- Исправляйте ошибки в валидаторе - они улучшают RAG и active learning
- Запустите `python quickstart.py` для проверки
- Соберите feedback: `curl http://localhost:8000/api/active-learning/stats`

---

## Команды для быстрого старта

```bash
# Установка
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# Web UI
python -m avers web --port 8000
# http://localhost:8000
# http://localhost:8000/annotator

# Датасет
python -m avers dataset generate --output /tmp/dataset --num-train 100 --num-val 20
python -m avers dataset preview --output /tmp/preview --num 5

# Обучение (нужен GPU + ultralytics)
pip install ultralytics
python -m avers dataset train --data /tmp/dataset/dataset.yaml --model rtdetr-l --epochs 10

# RAG
python -m avers rag stats
python quickstart.py

# Тесты
python -m pytest tests/ -v
```
