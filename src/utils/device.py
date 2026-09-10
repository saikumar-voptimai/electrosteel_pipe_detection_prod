from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeDevice:
    requested: int | str | None
    model_path: str
    model_suffix: str
    torch_available: bool
    torch_cuda_available: bool
    torch_version: str | None
    cuda_device_name: str | None
    torch_device: str
    ultralytics_device: int | str | None
    half: bool

    @property
    def is_cuda(self) -> bool:
        return self.torch_device.startswith("cuda")

    @property
    def is_tensorrt(self) -> bool:
        return self.model_suffix == ".engine"

    @property
    def is_onnx(self) -> bool:
        return self.model_suffix == ".onnx"


@lru_cache(maxsize=1)
def _torch_module() -> Any | None:
    try:
        import torch

        return torch
    except Exception as exc:
        logger.warning("PyTorch import failed; falling back to CPU runtime | error=%s", exc)
        return None


def _normalize_requested(requested: int | str | None) -> int | str | None:
    if requested is None:
        return None
    if isinstance(requested, int):
        return requested

    normalized = str(requested).strip().lower()
    if normalized in ("", "none", "default"):
        return None
    if normalized in ("auto", "cpu", "cuda", "cuda:0"):
        return normalized
    if normalized.isdigit():
        return int(normalized)
    return str(requested).strip()


def select_runtime_device(
    requested: int | str | None = "auto",
    model_path: str = "",
    half: bool = False,
) -> RuntimeDevice:
    """
    Resolve one runtime device policy for the application.

    Ultralytics accepts integer CUDA ids such as 0, while torch modules prefer
    "cuda:0". This helper keeps both forms together and provides a CPU fallback
    when PyTorch was installed without CUDA support.
    """
    requested_norm = _normalize_requested(requested)
    suffix = Path(str(model_path)).suffix.lower()
    torch = _torch_module()
    torch_available = torch is not None
    torch_cuda_available = False
    torch_version = None
    cuda_device_name = None

    if torch is not None:
        torch_version = getattr(torch, "__version__", None)

    if suffix == ".engine":
        ultralytics_device: int | str | None = 0 if requested_norm != "cpu" else "cpu"
        torch_device = "cuda:0" if ultralytics_device != "cpu" else "cpu"
        use_half = False
        return RuntimeDevice(
            requested=requested,
            model_path=str(model_path),
            model_suffix=suffix,
            torch_available=torch_available,
            torch_cuda_available=False,
            torch_version=torch_version,
            cuda_device_name=None,
            torch_device=torch_device,
            ultralytics_device=ultralytics_device,
            half=use_half,
        )

    if torch is not None:
        try:
            torch_cuda_available = bool(torch.cuda.is_available())
            if torch_cuda_available:
                cuda_device_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            logger.warning("CUDA availability check failed; falling back to CPU | error=%s", exc)
            torch_cuda_available = False

    wants_auto = requested_norm in (None, "auto")
    wants_cuda = (
        requested_norm == "cuda"
        or requested_norm == "cuda:0"
        or isinstance(requested_norm, int)
    )

    if wants_auto:
        ultralytics_device = 0 if torch_cuda_available else "cpu"
        torch_device = "cuda:0" if torch_cuda_available else "cpu"
    elif wants_cuda:
        if torch_cuda_available:
            cuda_id = requested_norm if isinstance(requested_norm, int) else 0
            ultralytics_device = cuda_id
            torch_device = f"cuda:{cuda_id}"
        else:
            logger.warning(
                "CUDA was requested but this PyTorch build has no CUDA; falling back to CPU."
            )
            ultralytics_device = "cpu"
            torch_device = "cpu"
    elif requested_norm == "cpu":
        ultralytics_device = "cpu"
        torch_device = "cpu"
    else:
        ultralytics_device = requested_norm
        torch_device = str(requested_norm)

    use_half = bool(half and torch_device.startswith("cuda") and suffix not in (".engine", ".onnx"))
    return RuntimeDevice(
        requested=requested,
        model_path=str(model_path),
        model_suffix=suffix,
        torch_available=torch_available,
        torch_cuda_available=torch_cuda_available,
        torch_version=torch_version,
        cuda_device_name=cuda_device_name,
        torch_device=torch_device,
        ultralytics_device=ultralytics_device,
        half=use_half,
    )


def configure_torch_backend(device: RuntimeDevice) -> None:
    torch = _torch_module()
    if torch is None or not device.is_cuda:
        return

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        logger.debug("Could not enable torch.backends.cudnn.benchmark", exc_info=True)

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        logger.debug("Could not enable TF32 backend flags", exc_info=True)
