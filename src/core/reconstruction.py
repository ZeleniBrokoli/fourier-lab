from __future__ import annotations

import numpy as np

from core.fft_utils import fft2_shift, ifft2_shift


def reconstruct_from_top_coefficients(
    image_gray: np.ndarray,
    keep_ratio: float,
) -> np.ndarray:
    """
    Reconstruct image using only the strongest FFT coefficients.

    keep_ratio:
      0.0 -> almost nothing
      1.0 -> full reconstruction
    """
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))

    spectrum = fft2_shift(image_gray)
    flat_magnitude = np.abs(spectrum).ravel()

    total = flat_magnitude.size
    keep_count = max(1, int(total * keep_ratio))

    if keep_count >= total:
        return ifft2_shift(spectrum)

    idx = np.argpartition(flat_magnitude, -keep_count)[-keep_count:]
    mask = np.zeros(total, dtype=np.float32)
    mask[idx] = 1.0
    mask = mask.reshape(spectrum.shape)

    filtered = spectrum * mask
    return ifft2_shift(filtered)


def progressive_reconstruction_frames(
    image_gray: np.ndarray,
    steps: int = 20,
) -> list[tuple[float, np.ndarray]]:
    """Generate reconstruction frames from low to high keep ratios."""
    steps = max(2, int(steps))
    ratios = np.linspace(0.01, 1.0, steps)

    frames: list[tuple[float, np.ndarray]] = []
    for ratio in ratios:
        recon = reconstruct_from_top_coefficients(image_gray, float(ratio))
        frames.append((float(ratio), recon))
    return frames