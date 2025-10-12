import subprocess, json, tempfile
from typing import NamedTuple


class MediaInfo(NamedTuple):
    mime: str | None
    size: int | None
    duration_s: float | None
    width: int | None
    height: int | None


def ffprobe_info(path: str) -> MediaInfo:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path
        ], stderr=subprocess.STDOUT)
        info = json.loads(out)
        fmt = info.get("format", {})
        streams = info.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
        return MediaInfo(
            mime=v.get("codec_name"),
            size=int(float(fmt.get("size", 0))) if fmt.get("size") else None,
            duration_s=float(fmt.get("duration")) if fmt.get("duration") else None,
            width=v.get("width"),
            height=v.get("height"),
        )
    except Exception:
        return MediaInfo(None, None, None, None, None)