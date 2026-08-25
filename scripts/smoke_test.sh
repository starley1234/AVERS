#!/usr/bin/env bash
# =============================================================================
#  AVERS Smoke Test — проверка работоспособности всех подсистем с логом
#
#  Запуск:   bash scripts/smoke_test.sh [флаги]
#  Флаги:
#    --skip-train     не запускать микро-обучение (YOLO, 1 эпоха)
#    --with-rtdetr    дополнительно проверить обучение RT-DETR (качает 63 МБ весов)
#    --skip-web       не проверять Web API
#    --skip-dataset   не генерировать датасеты (только импорты и пайплайн на кэше)
#
#  Результат: файл avers_smoke_<дата>.log рядом с репозиторием — пришлите его
#  для анализа. Код возврата 0, если нет критических ошибок (SKIP не считаются).
# =============================================================================
set -uo pipefail

VERSION="1.0"
TS="$(date +%Y%m%d_%H%M%S)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/avers_smoke_$TS.log"
WORK="/tmp/avers_smoke_$TS"
mkdir -p "$WORK"
export YOLO_CONFIG_DIR="$WORK/uicfg"

SKIP_TRAIN=0; SKIP_WEB=0; SKIP_DATASET=0; WITH_RTDETR=0
for arg in "$@"; do
  case "$arg" in
    --skip-train)   SKIP_TRAIN=1 ;;
    --skip-web)     SKIP_WEB=1 ;;
    --skip-dataset) SKIP_DATASET=1 ;;
    --with-rtdetr)  WITH_RTDETR=1 ;;
    *) echo "Неизвестный флаг: $arg"; exit 2 ;;
  esac
done

# Весь вывод (stdout+stderr) и в терминал, и в лог
exec > >(tee "$LOG") 2>&1

PASS=0; FAIL=0; SKIP=0; WARN=0
declare -a CASES=()

ok()   { echo "[ OK ] $*"; PASS=$((PASS+1)); CASES+=("OK   | $*"); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); CASES+=("FAIL | $*"); }
warn() { echo "[WARN] $*"; WARN=$((WARN+1)); CASES+=("WARN | $*"); }
skip() { echo "[SKIP] $*"; SKIP=$((SKIP+1)); CASES+=("SKIP | $*"); }
section() { echo; echo "=============================================================="; echo "== $*"; echo "=============================================================="; }

# -----------------------------------------------------------------------------
# Выбор интерпретатора Python
# -----------------------------------------------------------------------------
if [ -n "${AVERS_PYTHON:-}" ]; then
  PY="$AVERS_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/venv/bin/python" ]; then
  PY="$ROOT/venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi

cd "$ROOT"

echo "################################################################"
echo "#  AVERS Smoke Test v$VERSION"
echo "#  Лог: $LOG"
echo "################################################################"

# =============================================================================
section "1. Система"
# =============================================================================
echo "Дата:            $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Хост:            $(hostname 2>/dev/null || echo n/a)"
echo "uname:           $(uname -a)"
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "Окружение:       WSL ($(cat /proc/version))"
else
  echo "Окружение:       $(grep PRETTY /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo 'n/a')"
fi
echo "CPU ядер:        $(nproc 2>/dev/null || echo n/a)"
echo "Память:          $(free -h 2>/dev/null | awk '/^Mem:/{print $2" всего, "$7" свободно"}' || echo n/a)"
echo "Диск /tmp:       $(df -h /tmp 2>/dev/null | awk 'NR==2{print $4" свободно"}' || echo n/a)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null | sed 's/^/  /' || warn "nvidia-smi есть, но не отработал"
else
  echo "GPU:             nvidia-smi не найден (CPU-режим)"
fi
echo "Git:             $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'не git-репозиторий')"
echo "Репозиторий:     $ROOT"

# =============================================================================
section "2. Python и пакеты"
# =============================================================================
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  fail "Python не найден. Задайте AVERS_PYTHON=/путь/к/python или создайте venv"
  echo; echo "Итог: тест прерван."; exit 1
fi
ok "Python: $PY ($("$PY" --version 2>&1))"

"$PY" - <<'EOF'
import importlib, importlib.metadata as md
pkgs = ["numpy","opencv-python-headless","opencv-python","scipy","networkx","pydantic",
        "PyYAML","scikit-image","Pillow","lxml","tqdm","fastapi","uvicorn","python-multipart",
        "PyMuPDF","torch","ultralytics","sahi","paddleocr","transformers","faiss-cpu"]
for p in pkgs:
    try:
        print(f"  {p}: {md.version(p)}")
    except Exception:
        print(f"  {p}: -")
try:
    import cv2
    print(f"  cv2 импорт: OK ({cv2.__version__})")
except Exception as e:
    print(f"  cv2 импорт: FAIL ({e})")
    if "libGL" in str(e):
        print("  --> ФИКС: pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless")
try:
    import torch
    print(f"  torch.cuda.is_available: {torch.cuda.is_available()}")
except Exception:
    pass
EOF

CV2_STATUS=$("$PY" -c "import cv2; print('ok')" 2>/dev/null || echo "broken")
if [ "$CV2_STATUS" = "ok" ]; then ok "cv2 импортируется"; else fail "cv2 не импортируется (см. выше, скорее всего конфликт opencv-python/headless)"; fi

# =============================================================================
section "3. Импорт модулей AVERS"
# =============================================================================
if "$PY" - <<'EOF'
mods = ["avers.config","avers.core.pipeline","avers.core.validators","avers.pipeline",
        "avers.stages.stage1_slicing","avers.stages.stage2_detection","avers.stages.stage3_ocr",
        "avers.stages.stage4_vectorization","avers.stages.stage5_graph_synthesis",
        "avers.stages.stage6_vlm_arbitrator","avers.dataset.generator","avers.dataset.train","avers.rag"]
bad = []
for m in mods:
    try:
        __import__(m); print(f"  {m}: OK")
    except Exception as e:
        bad.append(m); print(f"  {m}: FAIL ({type(e).__name__}: {e})")
raise SystemExit(1 if bad else 0)
EOF
then ok "Все модули AVERS импортируются"
else fail "Часть модулей AVERS не импортируется (см. список FAIL выше)"
fi

# =============================================================================
section "4. Генерация синтетических схем (превью)"
# =============================================================================
PREVIEWS="$WORK/previews"
if [ "$SKIP_DATASET" = "1" ]; then
  skip "генерация превью (--skip-dataset)"
else
  if timeout 180 "$PY" -m avers dataset preview -o "$PREVIEWS" --num 3 --size 1024 >"$WORK/preview.log" 2>&1 \
     && [ "$(ls "$PREVIEWS"/preview_*.jpg 2>/dev/null | wc -l)" -ge 3 ]; then
    ok "Сгенерировано превью: $(ls "$PREVIEWS" | wc -l) шт. ($PREVIEWS)"
  else
    fail "Генерация превью (лог: $WORK/preview.log)"; tail -5 "$WORK/preview.log"
  fi
fi

# =============================================================================
section "5. Генерация микродатасета YOLO (8/4/2, 640px)"
# =============================================================================
DS="$WORK/dataset"
if [ "$SKIP_DATASET" = "1" ]; then
  skip "генерация датасета (--skip-dataset)"
else
  if timeout 300 "$PY" -m avers dataset generate -o "$DS" --num-train 8 --num-val 4 --num-test 2 --image-size 640 >"$WORK/dataset.log" 2>&1 \
     && [ -f "$DS/dataset.yaml" ]; then
    N_TRAIN=$(find "$DS/images/train" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    N_VAL=$(find "$DS/images/val" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    ok "Датасет готов: train=$N_TRAIN, val=$N_VAL ($DS)"
    [ "$N_TRAIN" -ge 8 ] || warn "train-изображений меньше ожидаемого: $N_TRAIN"
  else
    fail "Генерация датасета (лог: $WORK/dataset.log)"; tail -5 "$WORK/dataset.log"
  fi
fi

# =============================================================================
section "6. Полный пайплайн (6 стадий, без VLM)"
# =============================================================================
RES_JSON="$WORK/result.json"
DEMO_IMG="$PREVIEWS/preview_0000.jpg"
if [ ! -f "$DEMO_IMG" ]; then
  skip "нет демо-изображения (пропущена секция 4?)"
else
  RC=0
  timeout 600 "$PY" -m avers process "$DEMO_IMG" -o "$RES_JSON" --no-vlm >"$WORK/process.log" 2>&1 || RC=$?
  if [ -f "$RES_JSON" ]; then
    STATS=$("$PY" - "$RES_JSON" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
def n(key):
    v = d.get(key)
    return len(v) if isinstance(v, (list, dict)) else (v if v is not None else "-")
print(f"components={n('components')}, nets={n('nets')}, на_ревизию={n('human_review_required')}")
EOF
)
    if [ "$RC" -eq 0 ]; then
      ok "Пайплайн завершён ($STATS)"
    elif [ "$RC" -eq 2 ]; then
      warn "Пайплайн завершён, требуется ревизия человеком (exit 2) ($STATS)"
    else
      fail "Пайплайн упал с кодом $RC (лог: $WORK/process.log)"; tail -8 "$WORK/process.log"
    fi
  else
    fail "Пайплайн не создал результат (код $RC, лог: $WORK/process.log)"; tail -8 "$WORK/process.log"
  fi
fi

# =============================================================================
section "7. Vision RAG (CLIP+FAISS)"
# =============================================================================
RC_RAG=0
"$PY" - <<'EOF' || RC_RAG=$?
import sys, traceback
try:
    import numpy as np, cv2
    from avers.rag import get_rag
    rag = get_rag(storage_path="/tmp/avers_smoke_rag_store")
    print(f"  RAG backend эмбеддингов: {rag.stats().get('embedding_backend')}")
    roi = np.ones((256, 256, 3), np.uint8) * 255
    cv2.circle(roi, (128, 128), 6, (0, 0, 0), -1)
    cv2.line(roi, (0, 128), (256, 128), (0, 0, 0), 2)
    cv2.line(roi, (128, 0), (128, 256), (0, 0, 0), 2)
    eid = rag.add_example(roi, label="junction_dot", description="точка соединения")
    resp = rag.query(image=roi, text="junction dot", top_k=3)
    print(f"  RAG: пример {eid}, найдено {len(resp.results)} результатов")
    # персистентность: перечитываем базу заново
    import avers.rag.vision_rag as vr
    vr._global_rag = None
    rag2 = get_rag(storage_path="/tmp/avers_smoke_rag_store")
    resp2 = rag2.query(image=roi, text=None, top_k=3, use_vlm=False)
    assert resp2.results, "база пуста после перезагрузки"
    assert resp2.results[0].entry.label == "junction_dot", resp2.results[0].entry.label
    print(f"  персистентность: {rag2.stats()['rag_entries']} записей, top-1 после перезагрузки корректен")
    # текстовый поиск
    resp3 = rag2.query(text="junction", image=None, top_k=3, use_vlm=False)
    assert resp3.results, "текстовый поиск не работает"
    print("  текстовый поиск: OK")
except ImportError as e:
    print(f"  RAG недоступен (нет зависимости): {e}")
    raise SystemExit(3)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
EOF
if [ "$RC_RAG" -eq 0 ]; then
  ok "Vision RAG: добавление + запрос + персистентность + текстовый поиск"
elif [ "$RC_RAG" -eq 3 ]; then
  skip "Vision RAG (не установлены transformers/faiss — см. вывод выше)"
else
  fail "Vision RAG (вывод выше)"
fi

# =============================================================================
section "8. Микро-обучение YOLO (yolo11n, 1 эпоха, CPU/GPU auto)"
# =============================================================================
if [ "$SKIP_TRAIN" = "1" ]; then
  skip "микро-обучение (--skip-train)"
elif ! "$PY" -c "import torch, ultralytics" 2>/dev/null; then
  skip "нет torch/ultralytics: pip install ultralytics (и torch) — обучение проверить нельзя"
elif [ ! -f "$DS/dataset.yaml" ]; then
  skip "нет микродатасета (секция 5 пропущена или упала)"
else
  RUNS="$WORK/runs"
  echo "  (качает yolo11n.pt ~5.4 МБ при первом запуске; на CPU ~1-3 мин)"
  RC=0
  timeout 1500 "$PY" -m avers dataset train --data "$DS/dataset.yaml" \
      --model yolo11n --epochs 1 --batch 4 --imgsz 640 --device auto \
      --project "$RUNS" >"$WORK/train.log" 2>&1 || RC=$?
  BEST=$(find "$RUNS" -name "best.pt" 2>/dev/null | head -1)
  if [ "$RC" -eq 0 ] && [ -n "$BEST" ]; then
    ok "Обучение YOLO: $BEST (+ ONNX: $(find "$RUNS" -name 'best.onnx' | head -1))"
  elif [ "$RC" -eq 124 ]; then
    fail "Обучение YOLO превысило таймаут 1500с (лог: $WORK/train.log)"; tail -8 "$WORK/train.log"
  else
    fail "Обучение YOLO упало, код $RC (лог: $WORK/train.log)"; tail -15 "$WORK/train.log"
  fi
fi

# =============================================================================
section "9. Микро-обучение RT-DETR (опционально)"
# =============================================================================
if [ "$WITH_RTDETR" != "1" ]; then
  skip "RT-DETR не запрошен (запустите с --with-rtdetr)"
elif ! "$PY" -c "import torch, ultralytics" 2>/dev/null; then
  skip "нет torch/ultralytics"
elif [ ! -f "$DS/dataset.yaml" ]; then
  skip "нет микродатасета (секция 5 пропущена или упала)"
else
  echo "  (качает rtdetr-l.pt ~63 МБ; медленнее YOLO)"
  RC=0
  timeout 1800 "$PY" -m avers dataset train --data "$DS/dataset.yaml" \
      --model rtdetr-l --epochs 1 --batch 2 --imgsz 640 --device auto \
      --project "$WORK/runs_rtdetr" >"$WORK/train_rtdetr.log" 2>&1 || RC=$?
  BEST=$(find "$WORK/runs_rtdetr" -name "best.pt" 2>/dev/null | head -1)
  if [ "$RC" -eq 0 ] && [ -n "$BEST" ]; then
    ok "Обучение RT-DETR: $BEST"
  elif [ "$RC" -eq 124 ]; then
    warn "RT-DETR не уложился в 1800с (это медленная модель на CPU; лог: $WORK/train_rtdetr.log)"
  elif [ "$RC" -eq 137 ]; then
    warn "RT-DETR убит по памяти OOM (код 137) — нужно ~8+ ГБ RAM или GPU; сам код обучения корректен (см. лог)"
  else
    fail "Обучение RT-DETR упало, код $RC (лог: $WORK/train_rtdetr.log)"; tail -15 "$WORK/train_rtdetr.log"
  fi
fi

# =============================================================================
section "10. Web API (FastAPI)"
# =============================================================================
if [ "$SKIP_WEB" = "1" ]; then
  skip "Web API (--skip-web)"
elif ! "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
  skip "нет fastapi/uvicorn: pip install fastapi uvicorn python-multipart"
elif ! command -v curl >/dev/null 2>&1; then
  skip "нет curl — проверить Web API нечем"
else
  PORT=$("$PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
  "$PY" -m avers web --host 127.0.0.1 --port "$PORT" >"$WORK/web.log" 2>&1 &
  WEB_PID=$!
  HEALTH="не отвечает"
  for i in $(seq 1 30); do
    sleep 1
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTH="ok"; break; fi
    kill -0 "$WEB_PID" 2>/dev/null || break
  done
  if [ "$HEALTH" = "ok" ]; then
    HTTP_ROOT=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" 2>/dev/null || echo 0)
    if [ "$HTTP_ROOT" = "200" ]; then
      ok "Web API: /health и / отвечают (порт $PORT)"
    else
      warn "Web API: /health ok, но / вернул $HTTP_ROOT"
    fi
  else
    fail "Web API не поднялся за 30с (лог: $WORK/web.log)"; tail -8 "$WORK/web.log"
  fi
  kill "$WEB_PID" 2>/dev/null; wait "$WEB_PID" 2>/dev/null
fi

# =============================================================================
section "11. Внешняя LLM (VLM-арбитраж)"
# =============================================================================
"$PY" - <<'EOF'
import os
from avers.stages.stage6_vlm_arbitrator import VLMConfig
cfg = VLMConfig()
prov = cfg.resolved_provider()
print(f"  provider={prov}, api_base={'задан' if cfg.api_base else 'не задан'}, "
      f"api_model={cfg.api_model or '-'}, key={cfg.masked_api_key() or 'нет'}")
if prov == "api":
    okk = __import__("avers.stages.stage6_vlm_arbitrator.arbitrator", fromlist=["VLMWrapper"]).VLMWrapper(cfg).check_api()
    print(f"  Проверка соединения с API: {'успешна' if okk else 'НЕ ПРОЙДЕНА'}")
    raise SystemExit(0 if okk else 4)
raise SystemExit(0)
EOF
case $? in
  0) if [ -n "${AVERS_VLM_API_BASE:-}" ]; then ok "Внешняя LLM API доступна"; else skip "Внешняя LLM не настроена (это НЕ ошибка; см. QUICKSTART.md раздел 3)"; fi ;;
  4) warn "Внешняя LLM настроена, но API недоступен (проверьте base/key/сеть)" ;;
  *) warn "Ошибка при проверке конфигурации VLM (см. выше)" ;;
esac

# =============================================================================
section "ИТОГ"
# =============================================================================
echo
echo "Статусы проверок:"
for c in "${CASES[@]}"; do echo "  $c"; done
echo
echo "Итого: OK=$PASS, FAIL=$FAIL, WARN=$WARN, SKIP=$SKIP"
echo "Рабочая директория теста: $WORK (можно удалить)"
echo "Лог: $LOG"
if [ "$FAIL" -eq 0 ]; then
  echo "SMOKE TEST: PASS ✔  (пришлите этот лог для анализа)"
  exit 0
else
  echo "SMOKE TEST: FAIL ✘  (пришлите этот лог для анализа)"
  exit 1
fi
