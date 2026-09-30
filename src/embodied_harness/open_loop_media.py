"""Shared W1/W5 transport frames. Scoring stays in native pixel coordinates."""
from __future__ import annotations

from pathlib import Path

TRANSPORT_MAX_EDGE = 1280
TRANSPORT_JPEG_QUALITY = 85
TRANSPORT_MAX_BYTES = 10 * 1024 * 1024
IMAGE_NOTE = (
    "Public images are already attached as JPEG. The attachment may be downscaled "
    "for transport (longest edge 1280). task_summary.images width/height are the "
    "native pixel size; submit answers in those original coordinates. The only "
    "legal action is SUBMIT."
)


def write_transport_frame(source: Path, dest_dir: Path, index: int) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    from PIL import Image

    dest_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        rgb = img.convert("RGB")
    native = rgb.size
    width, height = native
    scale = min(1.0, TRANSPORT_MAX_EDGE / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    dest = dest_dir / f"{index:03d}.jpg"
    rgb.save(dest, format="JPEG", quality=TRANSPORT_JPEG_QUALITY, optimize=True)
    quality = TRANSPORT_JPEG_QUALITY
    while dest.stat().st_size > TRANSPORT_MAX_BYTES and quality > 40:
        quality -= 15
        rgb.save(dest, format="JPEG", quality=quality, optimize=True)
    if dest.stat().st_size > TRANSPORT_MAX_BYTES:
        raise RuntimeError(f"Transport frame still exceeds 10 MiB after JPEG: {dest}")
    return dest, native, rgb.size


def clear_transport_frames(dest_dir: Path) -> None:
    if not dest_dir.is_dir():
        return
    for path in dest_dir.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            path.unlink()
