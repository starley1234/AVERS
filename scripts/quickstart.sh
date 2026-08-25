#!/usr/bin/env bash
# =============================================================================
#  AVERS Quickstart — одним скриптом: окружение + демо + пайплайн
#
#  Запуск:  bash scripts/quickstart.sh [флаги]
#
#  Флаги:
#    --web           в конце запустить Web UI (http://localhost:8000)
#    --with-ml       дополнительно установить ML-зависимости (ultralytics, sahi)
#    --full          сгенерировать полный датасет (1000/200/100 картинок, долго)
#    --skip-install  не создавать/обновлять окружение (уже установлено)
#
#  Примеры:
#    bash scripts/quickstart.sh                # минимальный прогон
#    bash scripts/quickstart.sh --web          # прогон + веб-интерфейс
#    bash scripts/quickstart.sh --with-ml --full
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
DEMO="/tmp/avers_demo"

WEB=0; WITH_ML=0; FULL=0; SKIP_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --web) WEB=1 ;;
    --with-ml) WITH_ML=1 ;;
    --full) FULL=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    *) echo "Неизвестный флаг: $arg"; exit 2 ;;
  esac
done

info() { echo -e "\033[1;34m[ AVERS ]\033[0m $*"; }
good() { echo -e "\033[1;32m[  OK   ]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ ОШИБКА]\033[0m $*" >&2; }

cat <<'BANNER'

   ╔═══════════════════════════════════════════════════════════╗
   ║   А В Е Р С  —  Автоматическая векторизация схем          ║
   ║   Быстрый старт: окружение → демо-схема → пайплайн        ║
   ╚═══════════════════════════════════════════════════════════╝
BANNER

# -----------------------------------------------------------------------------
# 1. Python и виртуальное окружение
# -----------------------------------------------------------------------------
PY_SYSTEM="$(command -v python3 || true)"
if [ -z "$PY_SYSTEM" ]; then
  err "python3 не найден. Установите Python 3.11+ (в WSL: sudo apt install python3 python3-venv)"
  exit 1
fi

info "Python: $($PY_SYSTEM --version 2>&1), репозиторий: $ROOT"

# Выбор python: .venv репозитория > venv репозитория > активированный venv > системный
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
elif [ -x "$ROOT/venv/bin/python" ]; then
  PY="$ROOT/venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="$PY_SYSTEM"
fi

if [ "$SKIP_INSTALL" = "0" ]; then
  if [ ! -x "$VENV/bin/python" ]; then
    info "Создаю виртуальное окружение $VENV ..."
    "$PY_SYSTEM" -m venv "$VENV" || { err "Не удалось создать venv (в WSL/Ubuntu: sudo apt install python3-venv)"; exit 1; }
  else
    info "Виртуальное окружение уже есть: $VENV"
  fi
  PY="$VENV/bin/python"

  info "Устанавливаю зависимости (это может занять несколько минут)..."
  "$VENV/bin/pip" install --upgrade pip -q
  "$VENV/bin/pip" install -r "$ROOT/requirements.txt" -q || { err "pip install -r requirements.txt не удался"; exit 1; }
else
  info "--skip-install: использую $PY без изменений"
fi

# -----------------------------------------------------------------------------
# 2. Целостность OpenCV (opencv-python затирает headless-сборку)
# -----------------------------------------------------------------------------
fix_opencv() {
  if ! "$PY" -c "import cv2" >/dev/null 2>&1; then
    info "Чиню OpenCV (конфликт opencv-python / opencv-python-headless)..."
    "$PY" -m pip uninstall -y opencv-python opencv-python-headless -q 2>/dev/null || true
    "$PY" -m pip install --force-reinstall --no-deps opencv-python-headless -q
  fi
  "$PY" -c "import cv2" >/dev/null 2>&1 && good "OpenCV импортируется ($("$PY" -c 'import cv2; print(cv2.__version__)'))"
}
fix_opencv

# -----------------------------------------------------------------------------
# 3. Установка пакета (чтобы `python -m avers` работал из любой директории)
# -----------------------------------------------------------------------------
if [ "$SKIP_INSTALL" = "0" ]; then
  if ! "$PY" -c "import avers" >/dev/null 2>&1; then
    info "Ставлю пакет avers (editable)..."
    "$VENV/bin/pip" install -e "$ROOT" -q
  fi
fi
"$PY" -c "import avers" >/dev/null 2>&1 && good "Пакет avers доступен" \
  || { err "Пакет avers не импортируется. Запустите: $PY -m pip install -e $ROOT"; exit 1; }

# -----------------------------------------------------------------------------
# 4. ML-зависимости (опционально)
# -----------------------------------------------------------------------------
if [ "$WITH_ML" = "1" ]; then
  info "Ставлю ML-зависимости (ultralytics, sahi)... torch скачается автоматически"
  "$PY" -m pip install ultralytics sahi -q || warn_ml=1
  fix_opencv   # ultralytics тянет opencv-python — возвращаем headless
  [ -z "${warn_ml:-}" ] && good "ML-зависимости установлены" || err "ML-зависимости не установились (см. вывод выше), обучение будет недоступно"
fi

# -----------------------------------------------------------------------------
# 5. Демо: генерация синтетической схемы ГОСТ
# -----------------------------------------------------------------------------
info "Генерирую демонстрационную схему ГОСТ..."
rm -rf "$DEMO"
"$PY" -m avers dataset preview -o "$DEMO/previews" --num 2 --size 1024 | tail -2
good "Демо-схемы: $DEMO/previews/preview_0000.jpg, preview_0001.jpg"

# -----------------------------------------------------------------------------
# 6. Демо: полный пайплайн
# -----------------------------------------------------------------------------
info "Прогоняю полный пайплайн (6 стадий, без VLM)..."
RC=0
"$PY" -m avers process "$DEMO/previews/preview_0000.jpg" -o "$DEMO/result.json" --no-vlm | tail -4 || RC=$?
if [ "$RC" -ne 0 ] && [ "$RC" -ne 2 ]; then
  err "Пайплайн упал с кодом $RC. Прислать лог: bash scripts/smoke_test.sh"
  exit 1
fi
[ "$RC" -eq 2 ] && info "Пайплайн завершился с пометкой 'требуется ревизия' (это штатный код 2)"
"$PY" - "$DEMO/result.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[ AVERS ] Результат: components={len(d.get('components', []))}, "
      f"nets={len(d.get('nets', []))}, на_ревизию={len(d.get('human_review_required', []))}")
EOF
good "Результат распознавания: $DEMO/result.json"

# -----------------------------------------------------------------------------
# 7. Полный датасет (опционально)
# -----------------------------------------------------------------------------
if [ "$FULL" = "1" ]; then
  info "Генерирую ПОЛНЫЙ синтетический датасет (1000/200/100) — это долго..."
  "$PY" -m avers dataset generate -o /tmp/avers_dataset
  good "Датасет: /tmp/avers_dataset/dataset.yaml"
  info "Обучение (GPU): $PY -m avers dataset train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l --epochs 100 --batch 8"
else
  info "Датасет не генерирую (флаг --full не задан). Быстрая проверка обучения:"
  echo "         $PY -m avers dataset generate -o /tmp/avers_dataset --num-train 50 --num-val 10 --num-test 5 --image-size 640"
  echo "         $PY -m avers dataset train --data /tmp/avers_dataset/dataset.yaml --model yolo11n --epochs 1 --batch 4"
fi

# -----------------------------------------------------------------------------
# Итог
# -----------------------------------------------------------------------------
echo
good "Готово! Что дальше:"
echo "  • Распознать свою схему:   $PY -m avers process мой_скан.tif -o result.json"
echo "  • Веб-интерфейс:           $PY -m avers web   → http://localhost:8000"
echo "  • Проверка работоспособности (лог пришлите для анализа):"
echo "                             bash scripts/smoke_test.sh"
echo "  • Внешняя LLM (VLM-арбитраж): см. QUICKSTART.md, раздел 3"
echo
if [ "$WEB" = "1" ]; then
  info "Запускаю Web UI на http://localhost:8000 (Ctrl+C для остановки)..."
  cd "$ROOT"
  exec "$PY" -m avers web --host 0.0.0.0 --port 8000
fi
