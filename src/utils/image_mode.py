from __future__ import annotations


RGB_MODE = "rgb"
BLACK_AND_WHITE_MODE = "black_and_white"

_RGB_ALIASES = {"rgb", "rbg", "color", "colour", "bgr"}
_BLACK_AND_WHITE_ALIASES = {
    "black_and_white",
    "black_white",
    "bw",
    "gray",
    "grey",
    "grayscale",
    "greyscale",
    "gray_scale",
    "grey_scale",
    "mono",
    "monochrome",
}


def normalize_analysis_image_mode(mode: str | None) -> str:
    if mode is None:
        return RGB_MODE

    normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _RGB_ALIASES:
        return RGB_MODE
    if normalized in _BLACK_AND_WHITE_ALIASES:
        return BLACK_AND_WHITE_MODE

    valid = "rgb, color, colour, bgr, black_and_white, bw, gray, grey, grayscale, greyscale, mono, monochrome"
    raise ValueError(f"Invalid analysis_image_mode {mode!r}. Expected one of: {valid}")
