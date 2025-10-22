from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence
from src.config import settings

@dataclass
class FFmpegResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False

async def run_ffmpeg(args: Sequence[str]) -> FFmpegResult:
    """
    Запускає ffmpeg асинхронно.
    Повертає код завершення, stdout/stderr, та ознаку таймауту.
    Не кидає виключення при returncode != 0.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=settings.encode_timeout_s)
        return FFmpegResult(proc.returncode, stdout or b"", stderr or b"", timed_out=False)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = await proc.communicate()
        return FFmpegResult(proc.returncode if proc.returncode is not None else -9,
                            stdout or b"", stderr or b"", timed_out=True)

async def transcode(
    input_path: str,
    output_path: str,
    *,
    use_gpu: bool = False,
    crf: int = 23,
    preset: str | None = None,
    audio_bitrate: str = "128k",
    extra_args: Sequence[str] | None = None
) -> FFmpegResult:
    """
    Проста транскодувальна команда h264 з faststart.
    """
    vcodec = "h264_nvenc" if use_gpu else "libx264"
    # розумний дефолт, якщо не задано
    if preset is None:
        preset = "fast" if use_gpu else "ultrafast"

    args = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", vcodec, "-preset", preset, "-crf", str(crf),
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", audio_bitrate,
    ]
    if extra_args:
        args.extend(extra_args)
    args.append(output_path)

    return await run_ffmpeg(args)
