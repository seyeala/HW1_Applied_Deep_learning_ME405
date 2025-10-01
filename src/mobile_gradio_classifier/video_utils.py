"""Utility helpers for reading video frames with optional dependencies."""
from __future__ import annotations

from typing import Iterator

from PIL import Image

# Optional imports for video reading
try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:  # pragma: no cover - optional dependency
    import imageio  # type: ignore
    _HAS_IMAGEIO = True
except Exception:  # pragma: no cover - optional dependency
    imageio = None  # type: ignore
    _HAS_IMAGEIO = False


def _iter_video_frames_cv2(path: str, target_fps: float) -> Iterator[Image.Image]:
    if cv2 is None:  # pragma: no cover - safety check
        raise ImportError("OpenCV is not available")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video with OpenCV.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(target_fps, 1e-3))))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame)
        idx += 1
    cap.release()


def _iter_video_frames_imageio(path: str, target_fps: float) -> Iterator[Image.Image]:
    if imageio is None:  # pragma: no cover - safety check
        raise ImportError("imageio is not available")
    rdr = imageio.get_reader(path)
    try:
        meta = rdr.get_meta_data()
        src_fps = meta.get("fps", 30.0)
        step = max(1, int(round(src_fps / max(target_fps, 1e-3))))
        for i, frame in enumerate(rdr):
            if i % step == 0:
                yield Image.fromarray(frame)
    finally:
        rdr.close()


def iter_video_frames(path: str, target_fps: float) -> Iterator[Image.Image]:
    """Iterate over frames sampled from ``path`` at approximately ``target_fps``.

    Falls back from OpenCV to imageio depending on availability.
    """
    if _HAS_CV2:
        yield from _iter_video_frames_cv2(path, target_fps)
    elif _HAS_IMAGEIO:
        yield from _iter_video_frames_imageio(path, target_fps)
    else:
        raise ImportError("Please install opencv-python or imageio to enable video processing.")
