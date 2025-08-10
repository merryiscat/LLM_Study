# log/logger.py
import logging, os, json
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.jsonl")

class JsonLineFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "_extra"):
            base.update(record._extra)
        return json.dumps(base, ensure_ascii=False)

def get_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        logger.addHandler(fh)
    return logger
