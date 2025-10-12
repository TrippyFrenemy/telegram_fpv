from enum import Enum

class Mode(str, Enum):
    latest = "latest"
    backfill = "backfill"
    resume = "resume"