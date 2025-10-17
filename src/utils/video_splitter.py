import cv2, os
from pathlib import Path

def split_video(path: str, out_dir: str, segment_s: int = 5) -> list[str]:
    """Розрізає відео на segment_s секунд і повертає список шляхів сегментів."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames_per_seg = int(fps * segment_s)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    base = Path(path).stem
    out_paths = []
    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    seg_idx = 0
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frames_per_seg == 0:
            if writer:
                writer.release()
            seg_name = f"{base}_{seg_idx:04d}.mp4"
            seg_path = Path(out_dir) / seg_name
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(seg_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            out_paths.append(str(seg_path))
            seg_idx += 1

        writer.write(frame)
        frame_idx += 1

    if writer:
        writer.release()
    cap.release()
    return out_paths
