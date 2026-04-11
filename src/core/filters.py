from __future__ import annotations

import numpy as np

from core.fft_utils import fft2_shift, ifft2_shift


def circular_mask(shape: tuple[int, int], radius: int, mode: str = "low") -> np.ndarray:
    """
    Create a circular low-pass or high-pass mask.

    mode:
      - "low": keep frequencies inside radius
      - "high": keep frequencies outside radius
    """
    h, w = shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    low_mask = (dist <= radius).astype(np.float32)

    if mode == "low":
        return low_mask
    if mode == "high":
        return 1.0 - low_mask

    raise ValueError("mode must be 'low' or 'high'")


def apply_low_pass(image_gray: np.ndarray, radius: int) -> np.ndarray:
    """Low-pass filter in frequency domain."""
    spectrum = fft2_shift(image_gray)
    mask = circular_mask(spectrum.shape, radius, mode="low")
    filtered = spectrum * mask
    return ifft2_shift(filtered)


def apply_high_pass(image_gray: np.ndarray, radius: int) -> np.ndarray:
    """High-pass filter in frequency domain."""
    spectrum = fft2_shift(image_gray)
    mask = circular_mask(spectrum.shape, radius, mode="high")
    filtered = spectrum * mask
    return ifft2_shift(filtered)


def apply_notch_filter(
    image_gray: np.ndarray,
    centers: list[tuple[int, int]],
    notch_radius: int = 8,
) -> np.ndarray:
    """
    Remove specific frequency blobs around selected centers.
    centers are given in shifted spectrum coordinates.
    """
    spectrum = fft2_shift(image_gray)
    h, w = spectrum.shape
    mask = np.ones((h, w), dtype=np.float32)

    yy, xx = np.ogrid[:h, :w]
    for cy, cx in centers:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask[dist <= notch_radius] = 0.0

    filtered = spectrum * mask
    return ifft2_shift(filtered)