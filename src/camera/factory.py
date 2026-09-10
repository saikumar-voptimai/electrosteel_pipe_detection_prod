from __future__ import annotations

from typing import Any

from camera.clients import BaslerCameraClient, CameraClient, OpenCVCameraClient, VAImagingCameraClient
from utils.config import CameraCfg


def resolve_camera_type(camera_cfg: CameraCfg | None, source: int | str | None = None) -> str:
    if isinstance(source, int):
        return "opencv"
    if isinstance(source, str):
        source_name = source.strip().lower()
        if source_name.startswith("gige"):
            return (camera_cfg.type.strip().lower() if camera_cfg and camera_cfg.type else "va_imaging")
        if source_name in ("basler", "va_imaging", "camera"):
            return (camera_cfg.type.strip().lower() if camera_cfg and camera_cfg.type else source_name)
        if source_name:
            return "opencv"
    if camera_cfg and camera_cfg.type:
        return camera_cfg.type.strip().lower()
    return "opencv"


def create_camera_client(source: int | str, camera_cfg: CameraCfg | None = None, **kwargs: Any) -> CameraClient:
    camera_type = resolve_camera_type(camera_cfg, source)
    if camera_type == "opencv":
        return OpenCVCameraClient(source=source)
    if camera_type == "va_imaging":
        if camera_cfg is None:
            raise ValueError("VA Imaging camera selected but camera config is missing")
        return VAImagingCameraClient(source=source, camera_cfg=camera_cfg, **kwargs)
    if camera_type == "basler":
        if camera_cfg is None:
            raise ValueError("Basler camera selected but camera config is missing")
        return BaslerCameraClient(source=source, camera_cfg=camera_cfg)
    raise ValueError(
        f"Unsupported camera type {camera_type!r}. "
        "Expected one of: va_imaging, basler, opencv."
    )
