from pathlib import Path
from src.config import settings


root = settings.fs_root
root.mkdir(parents=True, exist_ok=True)
