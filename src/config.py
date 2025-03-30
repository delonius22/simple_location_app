import os

# Database settings
DB_PATH = os.environ.get("DB_PATH", "login_anomaly_detection.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Application settings
DEFAULT_ANOMALY_THRESHOLD = 0.7
CACHE_MAX_SIZE = 1000

