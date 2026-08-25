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

## 🔄 v0.3 - Active Learning Loop (Next - In Progress)
- [x] ActiveLearningLoop base (done)
- [x] Web UI integration for feedback (done - resolve_issue now adds to loop + RAG)
- [ ] Auto retraining scheduler (cron/background job)
- [ ] Model versioning and A/B testing
- [ ] Metrics dashboard (accuracy improvement over time)
- [ ] Notification when retrain needed
- [ ] Merge synthetic + feedback datasets for training

**CLI:**
```bash
python -m avers web --port 8000  # validator now auto-collects feedback
curl http://localhost:8000/api/active-learning/stats
curl -X POST http://localhost:8000/api/active-learning/retrain?model_type=rtdetr-l&epochs=20
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

### Высокий приоритет (для MVP)
1. **Собрать реальный датасет** - отсканировать 20-50 схем БКС, разметить через /annotator
2. **Обучить модель** - запустить `avers dataset generate` + `avers dataset train` на реальном GPU
3. **Протестировать на реальных сканах** - прогнать через pipeline, собрать feedback
4. **Настроить Active Learning** - подключить retrain scheduler

### Средний приоритет
5. **Docker production** - протестировать docker-compose с GPU
6. **Экспорт в САПР** - исследовать форматы Max-САПР, КОМПАС
7. **Улучшить OCR** - дообучить PaddleOCR на ГОСТ шрифтах (ГОСТ 2.304)
8. **Добавить больше ГОСТ символов** - реле, контакторы, etc.

### Низкий приоритет (nice to have)
9. **SAM integration** для аннотатора
10. **Qdrant** вместо FAISS
11. **K8s deployment**

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
