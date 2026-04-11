from __future__ import annotations

import numpy as np
from PIL import Image


def load_image(uploaded_file) -> np.ndarray:
    """Load uploaded image as RGB uint8 array."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image to grayscale float32 array."""
    if image.ndim == 2:
        return image.astype(np.float32)

    rgb = image[..., :3].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def resize_max(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    """Resize image while preserving aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image

    scale = max_side / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    pil_image = Image.fromarray(image.astype(np.uint8))
    resized = pil_image.resize((new_w, new_h))
    return np.array(resized)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize any numeric image to uint8 [0, 255] for display."""
    img = np.real(image).astype(np.float32)
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    if np.isclose(min_val, max_val):
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = np.clip(img * 255.0, 0, 255)
    return img.astype(np.uint8)


def fft2_shift(image_gray: np.ndarray) -> np.ndarray:
    """Centered 2D FFT."""
    image_gray = image_gray.astype(np.float32)
    return np.fft.fftshift(np.fft.fft2(image_gray))


def ifft2_shift(spectrum_shifted: np.ndarray) -> np.ndarray:
    """Inverse centered 2D FFT, returning real-valued image."""
    image = np.fft.ifft2(np.fft.ifftshift(spectrum_shifted))
    return np.real(image)


def spectrum_magnitude(spectrum_shifted: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrum normalized for display."""
    magnitude = np.log1p(np.abs(spectrum_shifted))
    return normalize_image(magnitude)


def split_rgb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split RGB image into channels."""
    rgb = image[..., :3]
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]