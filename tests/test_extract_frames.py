"""Tests para debug/extract_frames.py: extracción por timestamps/intervalo y recorte de ROIs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.types import ROI
from debug.extract_frames import (
    crop_roi_from_image,
    extract_frames_by_interval,
    extract_frames_by_timestamps,
)


def _create_synthetic_video(path: Path, width: int = 320, height: int = 180, fps: float = 10.0, frames: int = 20) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            img = np.full((height, width, 3), fill_value=(idx * 10) % 255, dtype=np.uint8)
            writer.write(img)
    finally:
        writer.release()


def test_extract_frames_by_timestamps_and_interval(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _create_synthetic_video(video_path, width=320, height=180, fps=10.0, frames=20)

    out_ts = tmp_path / "by_ts"
    saved_ts = extract_frames_by_timestamps(
        video_path=video_path,
        output_dir=out_ts,
        timestamps=[0.0, 0.5, 1.0],
        expected_size=(320, 180),
    )
    assert len(saved_ts) == 3
    img = cv2.imread(str(saved_ts[0]), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (180, 320, 3)

    out_int = tmp_path / "by_interval"
    saved_int = extract_frames_by_interval(
        video_path=video_path,
        output_dir=out_int,
        interval_sec=0.5,
        start_sec=0.0,
        end_sec=1.5,
    )
    assert len(saved_int) == 4


def test_crop_roi_from_image_and_preview(tmp_path: Path) -> None:
    src = tmp_path / "frame.png"
    canvas = np.zeros((200, 300, 3), dtype=np.uint8)
    canvas[40:90, 50:150] = (10, 200, 30)
    cv2.imwrite(str(src), canvas)

    crop_out = tmp_path / "templates" / "ui" / "fight.png"
    preview_out = tmp_path / "preview.png"
    result_path = crop_roi_from_image(
        image_path=src,
        roi=ROI(x=50, y=40, width=100, height=50),
        output_path=crop_out,
        preview_path=preview_out,
        label="fight",
    )

    assert result_path == crop_out
    cropped = cv2.imread(str(crop_out), cv2.IMREAD_COLOR)
    assert cropped is not None
    assert cropped.shape == (50, 100, 3)
    assert tuple(cropped[0, 0]) == (10, 200, 30)
    assert preview_out.is_file()


def test_crop_roi_out_of_bounds_raises(tmp_path: Path) -> None:
    src = tmp_path / "frame.png"
    cv2.imwrite(str(src), np.zeros((100, 100, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="fuera de los límites"):
        crop_roi_from_image(
            image_path=src,
            roi=ROI(x=80, y=80, width=50, height=50),
            output_path=tmp_path / "bad.png",
        )
