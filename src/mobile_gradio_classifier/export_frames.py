"""Utilities for exporting frames from a batch of video files."""
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import yaml
from PIL import Image

from .video_utils import iter_video_frames

try:  # Pillow>=9
    _RESAMPLE = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback for older Pillow
    _RESAMPLE = Image.LANCZOS


@dataclass
class FrameExportConfig:
    """Configuration for exporting frames from videos."""

    input_glob: Sequence[str]
    output_dir: Path
    size: Tuple[int, int]
    format: str
    fps: float
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.input_glob = tuple(self.input_glob)
        self.output_dir = Path(self.output_dir)
        self.size = (int(self.size[0]), int(self.size[1]))
        self.format = str(self.format)
        self.fps = float(self.fps)


def _ensure_glob_sequence(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        globs: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Each glob pattern must be a string.")
            globs.append(item)
        if not globs:
            raise ValueError("At least one glob pattern must be provided.")
        return tuple(globs)
    raise TypeError("`input_glob` must be a string or a sequence of strings.")


def _ensure_size(value: object) -> Tuple[int, int]:
    if not isinstance(value, (Sequence, tuple, list)) or isinstance(value, (str, bytes)):
        raise TypeError("`size` must be a sequence of two positive integers.")
    if len(value) != 2:  # type: ignore[arg-type]
        raise ValueError("`size` must contain exactly two values: width and height.")
    width, height = value  # type: ignore[misc]
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise TypeError("`size` values must be numeric.")
    width_i, height_i = int(width), int(height)
    if width_i <= 0 or height_i <= 0:
        raise ValueError("`size` values must be positive.")
    return width_i, height_i


def load_config(path: str) -> FrameExportConfig:
    """Load and validate a :class:`FrameExportConfig` from ``path``."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping of options.")

    required_keys = {"input_glob", "output_dir", "size", "format", "fps"}
    missing = required_keys.difference(data)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required configuration keys: {missing_list}")

    input_glob = _ensure_glob_sequence(data["input_glob"])
    size = _ensure_size(data["size"])

    output_dir = data["output_dir"]
    if not isinstance(output_dir, str):
        raise TypeError("`output_dir` must be a string path.")

    fmt = data["format"]
    if not isinstance(fmt, str) or not fmt:
        raise ValueError("`format` must be a non-empty string.")

    fps_value = data["fps"]
    if not isinstance(fps_value, (int, float)):
        raise TypeError("`fps` must be a numeric value.")
    fps = float(fps_value)
    if fps <= 0:
        raise ValueError("`fps` must be greater than zero.")

    overwrite = bool(data.get("overwrite", False))

    return FrameExportConfig(
        input_glob=input_glob,
        output_dir=Path(output_dir),
        size=size,
        format=fmt,
        fps=fps,
        overwrite=overwrite,
    )


def _expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        for match in sorted(glob(pattern)):
            path = Path(match)
            if path.is_file():
                paths.append(path)
    return paths


def main(config: FrameExportConfig) -> None:
    """Export frames for each video resolved from ``config``."""

    video_paths = _expand_inputs(config.input_glob)
    if not video_paths:
        raise FileNotFoundError("No videos matched the provided glob patterns.")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    extension = config.format.lower().lstrip(".")
    save_format = config.format.upper()

    for video_path in video_paths:
        video_stem = video_path.stem
        video_output_dir = config.output_dir / video_stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(iter_video_frames(str(video_path), config.fps)):
            frame_rgb = frame.convert("RGB")
            resized = frame_rgb.resize(config.size, resample=_RESAMPLE)
            filename = f"{video_stem}_{idx:06d}.{extension}"
            destination = video_output_dir / filename
            if destination.exists() and not config.overwrite:
                continue
            resized.save(destination, format=save_format)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export frames from videos based on a YAML config file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)


if __name__ == "__main__":
    cli()
