from dataclasses import dataclass
from rapidfuzz import fuzz
from src.config import settings


@dataclass
class FPVDecision:
    confidence: float
    reason: str

KEYS = None

def _keys():
    global KEYS
    if KEYS is None:
        KEYS = [k.strip() for k in settings.fpv_keywords.split(",") if k.strip()]
    return KEYS


def score(text: str | None, duration_s: float | None, width: int | None, height: int | None) -> FPVDecision:
    conf = 0.0
    reasons = []
    if text:
        mx = max([fuzz.partial_ratio(text.lower(), k.lower()) for k in _keys()] or [0]) / 100.0
        if mx > 0.6:
            conf += 0.3
            reasons.append(f"kw={mx:.2f}")
    if duration_s:
        if settings.fpv_min_duration_s <= duration_s <= settings.fpv_max_duration_s:
            conf += 0.3
            reasons.append("dur")
    if width and height:
        if height >= width: # часто портрет/вертикалка у FPV публікаціях
            conf += 0.1
            reasons.append("portrait")
    conf = min(1.0, conf)
    return FPVDecision(confidence=conf, reason=",".join(reasons))
