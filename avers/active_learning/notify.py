"""
Notifier — уведомления о событиях Active Learning Loop (v0.3).

Нужен, чтобы ответственный за дообучение человек (или другой сервис) узнавал,
что:
  - накопилось достаточно feedback для дообучения ("retrain_due"),
  - дообучение началось / завершилось / упало ("retrain_started" / "..._done" / "..._failed"),
  - новая модель продвинута в прод или откачена ("model_promoted" / "model_rolled_back").

Специально не тянет тяжёлые зависимости (requests, slack-sdk и т.п.) — хватает
`urllib` из стандартной библиотеки. Поддерживает:

  - лог (всегда, через avers.core.logger)
  - JSONL файл (для истории/дашборда, без БД)
  - generic webhook (Slack/Mattermost/n8n/Telegram-bridge/что угодно, что
    принимает POST JSON) — URL берётся из конфига или переменной окружения
    AVERS_NOTIFY_WEBHOOK_URL

Ошибки доставки webhook не должны ронять пайплайн — они логируются и
проглатываются.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from avers.core.logger import get_logger

logger = get_logger("avers.active_learning.notify")


class Notifier:
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        log_file: Optional[Path] = Path("/tmp/avers_feedback/notifications.jsonl"),
        timeout_sec: float = 5.0,
    ):
        self.webhook_url = webhook_url or os.environ.get("AVERS_NOTIFY_WEBHOOK_URL")
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec

    def notify(self, event_type: str, payload: Optional[Dict[str, Any]] = None, message: str = "") -> Dict[str, Any]:
        entry = {
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[notify] {event_type}: {message or payload}")

        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write notification log: {e}")

        if self.webhook_url:
            self._send_webhook(entry)

        return entry

    def _send_webhook(self, entry: Dict[str, Any]) -> None:
        try:
            body = json.dumps(
                {"text": f"[AVERS] {entry['event_type']}: {entry['message']}", **entry},
                ensure_ascii=False,
            ).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout_sec)
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning(f"Webhook notification failed: {e}")

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.log_file or not self.log_file.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        return entries[-limit:]


_global_notifier: Optional[Notifier] = None


def get_notifier(
    webhook_url: Optional[str] = None,
    log_file: Optional[Path] = Path("/tmp/avers_feedback/notifications.jsonl"),
) -> Notifier:
    global _global_notifier
    if _global_notifier is None:
        _global_notifier = Notifier(webhook_url=webhook_url, log_file=log_file)
    return _global_notifier
