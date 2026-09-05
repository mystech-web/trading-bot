"""Logging estructurado + alertas opcionales por email (SMTP) y/o Telegram.

Si no hay credenciales configuradas en `.env`, `send_alert` no rompe nada:
simplemente no manda nada por ningún canal y lo deja solo en el log local
(`reports/live.log`). Nunca debe tumbar la corrida del bot por un fallo de
red al mandar la alerta -- por eso todo está en try/except.
"""
from __future__ import annotations

import logging
import os
import smtplib
import urllib.parse
import urllib.request
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "reports"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "trading-bot", log_dir: Path | None = None) -> logging.Logger:
    """`log_dir` (default `reports/`): dónde escribir `live.log`. Pásale la
    subcarpeta del broker (ver `scripts/run_live_once.py`) para que el log de
    cada broker/perfil quede separado -- si no, acciones y cripto (o
    conservador y agresivo) terminan escribiendo el mismo archivo mezclado.
    El nombre del logger incluye la carpeta para que dos llamadas con el mismo
    `name` pero distinto `log_dir` no compartan por accidente el logger de
    Python cacheado (y con él, el archivo del primero)."""
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_key = f"{name}:{log_dir}"
    logger = logging.getLogger(logger_key)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_dir / "live.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    return logger


def _send_email(subject: str, body: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    if not host:
        return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("ALERT_EMAIL_TO", user)
    if not (user and password and to_addr):
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
    return True


def _send_telegram(subject: str, body: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return False

    text = f"*{subject}*\n{body}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status == 200


def send_alert(subject: str, body: str, logger: logging.Logger | None = None) -> None:
    logger = logger or get_logger()
    sent_any = False
    for sender, label in ((_send_email, "email"), (_send_telegram, "telegram")):
        try:
            if sender(subject, body):
                sent_any = True
                logger.info(f"Alerta enviada por {label}: {subject}")
        except Exception as e:
            logger.warning(f"Fallo enviando alerta por {label}: {e}")
    if not sent_any:
        logger.info(f"[Sin canal de alerta configurado] {subject}: {body}")
