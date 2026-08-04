from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Tuple

import numpy as np

from utils.config import BaslerCameraCfg, CameraCfg, CameraProfileCfg

logger = logging.getLogger(__name__)


FrameItem = Tuple[np.ndarray, float]


class CameraClient(Protocol):
    def open(self) -> None:
        ...

    def read(self) -> FrameItem | None:
        ...

    def close(self) -> None:
        ...

    def is_open(self) -> bool:
        ...

    def get_metadata(self) -> dict[str, Any]:
        ...


@dataclass
class OpenCVCameraClient:
    source: int | str
    _cap: object | None = field(default=None, init=False)

    def open(self) -> None:
        import cv2

        logger.info("Opening OpenCV video source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def read(self) -> FrameItem | None:
        if self._cap is None:
            self.open()
        ok, frame = self._cap.read()
        if ok and frame is not None:
            return frame, time.time()

        logger.info("Video source ended; restarting: %s", self.source)
        self._cap.set(0, 0)
        ok, frame = self._cap.read()
        if ok and frame is not None:
            return frame, time.time()
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def get_metadata(self) -> dict[str, Any]:
        return {"camera_type": "opencv", "source": self.source}


@dataclass
class VAImagingCameraClient:
    source: int | str
    camera_cfg: CameraCfg
    warmup_frames: int = 10
    _device_manager: object | None = field(default=None, init=False)
    _cam: object | None = field(default=None, init=False)
    _gx: object | None = field(default=None, init=False)

    def open(self) -> None:
        import gxipy as gx

        self._gx = gx
        logger.info("Opening VA Imaging camera")
        self._device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self._device_manager.update_all_device_list()
        if dev_num == 0:
            raise RuntimeError("No VA Imaging camera detected")

        device_info = self._resolve_device_info(dev_info_list)
        self._validate_control_access(device_info)
        sn = str(device_info["sn"])
        self._cam = self._device_manager.open_device_by_sn(sn)
        self._apply_base_camera_settings()
        self._cam.stream_on()

        for _ in range(max(0, int(self.warmup_frames))):
            img = self._cam.data_stream[0].get_image()
            if img is not None:
                img.convert("RGB")

    def _resolve_device_info(self, dev_info_list: list[dict[str, Any]]) -> dict[str, Any]:
        wanted_id = str(self.camera_cfg.id).strip()
        wanted_ip = str((self.camera_cfg.va_imaging or {}).get("ip", "")).strip()

        if wanted_id and wanted_id != "0":
            for info in dev_info_list:
                if wanted_id in self._device_identifiers(info):
                    info_ip = str(info.get("ip", "")).strip()
                    if wanted_ip and info_ip and wanted_ip != info_ip:
                        logger.warning(
                            "Configured VA Imaging camera id matched but configured ip differs | "
                            "id=%s configured_ip=%s detected_ip=%s",
                            wanted_id,
                            wanted_ip,
                            info_ip,
                        )
                    return info

        if wanted_ip:
            for info in dev_info_list:
                if wanted_ip == str(info.get("ip", "")).strip():
                    return info

        if wanted_id and wanted_id != "0" or wanted_ip:
            raise RuntimeError(
                "Configured VA Imaging camera was not detected. "
                f"id={wanted_id or '<unset>'}, ip={wanted_ip or '<unset>'}. "
                f"Detected cameras: {self._format_detected_devices(dev_info_list)}"
            )
        return dev_info_list[0]

    @staticmethod
    def _device_identifiers(device_info: dict[str, Any]) -> set[str]:
        return {
            str(device_info.get("sn", "")).strip(),
            str(device_info.get("ip", "")).strip(),
            str(device_info.get("mac", "")).strip(),
            str(device_info.get("user_id", "")).strip(),
            str(device_info.get("display_name", "")).strip(),
            str(device_info.get("model_name", "")).strip(),
            str(device_info.get("device_id", "")).strip(),
        }

    def _format_detected_devices(self, dev_info_list: list[dict[str, Any]]) -> str:
        return "; ".join(
            (
                f"sn={info.get('sn')}, ip={info.get('ip')}, user_id={info.get('user_id')}, "
                f"status={self._access_status_name(int(info.get('access_status', 0) or 0))}"
            )
            for info in dev_info_list
        )

    def _access_status_name(self, status: int) -> str:
        if self._gx is None:
            return str(status)
        return {
            int(self._gx.GxAccessStatus.UNKNOWN): "UNKNOWN",
            int(self._gx.GxAccessStatus.READWRITE): "READWRITE",
            int(self._gx.GxAccessStatus.READONLY): "READONLY",
            int(self._gx.GxAccessStatus.NOACCESS): "NOACCESS",
        }.get(status, str(status))

    def _validate_control_access(self, device_info: dict[str, Any]) -> None:
        if self._gx is None:
            return
        status = int(device_info.get("access_status", 0) or 0)
        if status == int(self._gx.GxAccessStatus.READWRITE):
            return
        status_name = self._access_status_name(status)
        if status == int(self._gx.GxAccessStatus.UNKNOWN):
            logger.warning(
                "VA Imaging camera access status is UNKNOWN; attempting open anyway | "
                "sn=%s ip=%s display_name=%s",
                device_info.get("sn"),
                device_info.get("ip"),
                device_info.get("display_name"),
            )
            return
        raise RuntimeError(
            "VA Imaging camera is not available for control access. "
            f"status={status_name}, sn={device_info.get('sn')}, ip={device_info.get('ip')}, "
            f"display_name={device_info.get('display_name')}. "
            "Close any Galaxy/Daheng viewer or other process using this camera, "
            "then power-cycle/reconnect the camera if it remains READONLY."
        )

    def _apply_base_camera_settings(self) -> None:
        if self._cam is None or self._gx is None:
            return
        cam = self._cam
        gx = self._gx
        cfg = self.camera_cfg
        try:
            logger.info("Applying VA Imaging base camera settings")
            if hasattr(cam, "ExposureAuto"):
                cam.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS if cfg.auto_exposure else gx.GxAutoEntry.OFF)
            if hasattr(cam, "GainAuto"):
                cam.GainAuto.set(gx.GxAutoEntry.CONTINUOUS if cfg.auto_gain else gx.GxAutoEntry.OFF)
            if hasattr(cam, "AcquisitionFrameRateMode"):
                cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
            if hasattr(cam, "AcquisitionFrameRate"):
                cam.AcquisitionFrameRate.set(cfg.fps)
        except Exception as exc:
            logger.warning("VA Imaging base camera configuration failed: %s", exc)

    def apply_camera_controls(self, *, auto_exposure: bool, auto_gain: bool) -> None:
        if self._cam is None or self._gx is None:
            return
        self.camera_cfg = replace(
            self.camera_cfg,
            auto_exposure=bool(auto_exposure),
            auto_gain=bool(auto_gain),
        )
        try:
            if hasattr(self._cam, "ExposureAuto"):
                self._cam.ExposureAuto.set(
                    self._gx.GxAutoEntry.CONTINUOUS if auto_exposure else self._gx.GxAutoEntry.OFF
                )
            if hasattr(self._cam, "GainAuto"):
                self._cam.GainAuto.set(
                    self._gx.GxAutoEntry.CONTINUOUS if auto_gain else self._gx.GxAutoEntry.OFF
                )
        except Exception as exc:
            logger.warning("VA Imaging auto exposure/gain apply failed: %s", exc)

    def apply_profile(self, profile: CameraProfileCfg | None) -> None:
        if profile is None or self._cam is None or self._gx is None:
            return
        cam = self._cam
        gx = self._gx
        try:
            logger.info("Applying VA Imaging camera profile | exposure=%s gain=%s", profile.exposure_us, profile.gain_db)
            if not self.camera_cfg.auto_exposure and hasattr(cam, "ExposureTime"):
                cam.ExposureTime.set(profile.exposure_us)
            if not self.camera_cfg.auto_gain and hasattr(cam, "Gain"):
                cam.Gain.set(profile.gain_db)
            if hasattr(cam, "GammaEnable"):
                cam.GammaEnable.set(bool(profile.gamma_enable))
                if profile.gamma_enable:
                    if hasattr(cam, "GammaMode"):
                        cam.GammaMode.set(gx.GxGammaModeEntry.USER)
                    if hasattr(cam, "Gamma"):
                        cam.Gamma.set(max(0.1, min(10.0, float(profile.gamma))))
        except Exception as exc:
            logger.warning("VA Imaging profile apply failed: %s", exc)

    def read(self) -> FrameItem | None:
        if self._cam is None:
            self.open()
        raw = self._cam.data_stream[0].get_image()
        if raw is None:
            return None
        rgb = raw.convert("RGB")
        img = rgb.get_numpy_array()
        if img is None:
            return None
        return np.ascontiguousarray(img[:, :, ::-1]), time.time()

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stream_off()
                self._cam.close_device()
            except Exception:
                logger.debug("VA Imaging close failed", exc_info=True)
            self._cam = None

    def is_open(self) -> bool:
        return self._cam is not None

    def get_metadata(self) -> dict[str, Any]:
        return {"camera_type": "va_imaging", "source": self.source, "id": self.camera_cfg.id}


@dataclass
class BaslerCameraClient:
    source: int | str
    camera_cfg: CameraCfg
    _camera: object | None = field(default=None, init=False)
    _converter: object | None = field(default=None, init=False)
    _pylon: object | None = field(default=None, init=False)

    def _import_pylon(self):
        try:
            from pypylon import pylon
        except ImportError as exc:
            raise RuntimeError(
                "Basler camera selected but pypylon is not installed. "
                "Install Basler pylon and then run: pip install pypylon"
            ) from exc
        return pylon

    @property
    def basler_cfg(self) -> BaslerCameraCfg:
        return self.camera_cfg.basler or BaslerCameraCfg()

    def open(self) -> None:
        pylon = self._import_pylon()
        self._pylon = pylon
        factory = pylon.TlFactory.GetInstance()
        device_info = self._select_device(factory)
        self._camera = pylon.InstantCamera(factory.CreateDevice(device_info))
        self._camera.Open()
        self._apply_grab_buffer_settings()
        self._apply_settings()
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._camera.StartGrabbing(self._grab_strategy())
        logger.info("Basler camera opened | metadata=%s", self.get_metadata())

    def _select_device(self, factory: Any) -> Any:
        devices = list(factory.EnumerateDevices())
        if not devices and self.basler_cfg.ip_address:
            devices = self._announce_remote_gige_device(factory)
        if not devices:
            raise RuntimeError("No Basler camera detected")

        cfg = self.basler_cfg
        for device in devices:
            serial = self._safe_device_value(device, "GetSerialNumber")
            user_id = self._safe_device_value(device, "GetUserDefinedName")
            ip_address = self._safe_device_value(device, "GetIpAddress")
            if cfg.serial_number and serial == cfg.serial_number:
                return device
            if cfg.device_user_id and user_id == cfg.device_user_id:
                return device
            if cfg.ip_address and ip_address == cfg.ip_address:
                return device

        if cfg.serial_number or cfg.device_user_id or cfg.ip_address:
            raise RuntimeError(
                "No Basler camera matched configured serial_number/device_user_id/ip_address "
                f"(serial_number={cfg.serial_number!r}, "
                f"device_user_id={cfg.device_user_id!r}, ip_address={cfg.ip_address!r})"
            )
        return devices[0]

    def _announce_remote_gige_device(self, factory: Any) -> list[Any]:
        for tl_info in factory.EnumerateTls():
            if self._safe_device_value(tl_info, "GetDeviceClass") != "BaslerGigE":
                continue
            transport = factory.CreateTl(tl_info)
            try:
                announced, device_info = transport.AnnounceRemoteDevice(self.basler_cfg.ip_address)
                if announced:
                    logger.info("Basler remote GigE device announced by IP: %s", self.basler_cfg.ip_address)
                    return [device_info]
            except Exception as exc:
                logger.debug("Basler remote announce failed | ip=%s | error=%s", self.basler_cfg.ip_address, exc)
        return []

    @staticmethod
    def _safe_device_value(device: Any, method_name: str) -> str:
        method = getattr(device, method_name, None)
        if not callable(method):
            return ""
        try:
            return str(method())
        except Exception:
            return ""

    def _apply_settings(self) -> None:
        cfg = self.basler_cfg
        if cfg.pixel_format:
            self._set_node("PixelFormat", cfg.pixel_format)
        if cfg.offset_x is not None:
            self._set_node("OffsetX", int(cfg.offset_x))
        if cfg.offset_y is not None:
            self._set_node("OffsetY", int(cfg.offset_y))
        if cfg.width is not None:
            self._set_node("Width", int(cfg.width))
        if cfg.height is not None:
            self._set_node("Height", int(cfg.height))
        if cfg.acquisition_frame_rate is not None:
            self._enable_frame_rate_control()
            self._set_node("AcquisitionFrameRate", float(cfg.acquisition_frame_rate))
        if cfg.exposure_time is not None:
            self._set_node("ExposureTime", float(cfg.exposure_time))
        if cfg.gain is not None:
            self._set_node("Gain", float(cfg.gain))
        if cfg.packet_size is not None:
            self._set_node("GevSCPSPacketSize", int(cfg.packet_size))
        if cfg.inter_packet_delay is not None:
            self._set_node("GevSCPD", int(cfg.inter_packet_delay))

    def _apply_grab_buffer_settings(self) -> None:
        if self._camera is None:
            return
        buffer_count = max(1, int(self.basler_cfg.max_num_buffer or 20))
        try:
            self._camera.MaxNumBuffer = buffer_count
        except Exception as exc:
            logger.warning("Basler buffer setting failed | MaxNumBuffer=%s | error=%s", buffer_count, exc)

    def _enable_frame_rate_control(self) -> None:
        for name, value in (
            ("AcquisitionFrameRateEnable", True),
            ("AcquisitionFrameRateEnabled", True),
            ("AcquisitionFrameRateMode", "On"),
        ):
            if self._set_node(name, value, warn_missing=False):
                return

    def _set_node(self, name: str, value: Any, warn_missing: bool = True) -> bool:
        try:
            node = self._camera.GetNodeMap().GetNode(name)
            if node is None:
                if warn_missing:
                    logger.warning("Basler setting unavailable: %s", name)
                return False
            if hasattr(node, "SetValue"):
                node.SetValue(value)
            else:
                setattr(self._camera, name, value)
            return True
        except Exception as exc:
            if warn_missing:
                logger.warning("Basler setting failed | %s=%s | error=%s", name, value, exc)
            return False

    def _grab_strategy(self) -> Any:
        pylon = self._pylon
        name = self.basler_cfg.grab_strategy.lower()
        strategies = {
            "latest_image_only": pylon.GrabStrategy_LatestImageOnly,
            "one_by_one": pylon.GrabStrategy_OneByOne,
            "latest_images": getattr(pylon, "GrabStrategy_LatestImages", pylon.GrabStrategy_LatestImageOnly),
        }
        if name not in strategies:
            raise ValueError(f"Unsupported Basler grab_strategy {self.basler_cfg.grab_strategy!r}")
        return strategies[name]

    def read(self) -> FrameItem | None:
        if self._camera is None:
            self.open()
        if not self._camera.IsGrabbing():
            self._camera.StartGrabbing(self._grab_strategy())

        grab_result = None
        try:
            grab_result = self._camera.RetrieveResult(
                int(self.basler_cfg.timeout_ms),
                self._pylon.TimeoutHandling_ThrowException,
            )
            if not grab_result.GrabSucceeded():
                logger.warning("Basler grab failed | code=%s | description=%s", grab_result.ErrorCode, grab_result.ErrorDescription)
                return None
            image = self._converter.Convert(grab_result)
            frame = image.GetArray()
            return np.ascontiguousarray(frame), time.time()
        finally:
            if grab_result is not None:
                grab_result.Release()

    def close(self) -> None:
        if self._camera is not None:
            try:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                if self._camera.IsOpen():
                    self._camera.Close()
            except Exception:
                logger.debug("Basler close failed", exc_info=True)
            self._camera = None

    def is_open(self) -> bool:
        return bool(self._camera is not None and self._camera.IsOpen())

    def get_metadata(self) -> dict[str, Any]:
        cfg = self.basler_cfg
        return {
            "camera_type": "basler",
            "serial_number": cfg.serial_number,
            "device_user_id": cfg.device_user_id,
            "ip_address": cfg.ip_address,
            "grab_strategy": cfg.grab_strategy,
        }
