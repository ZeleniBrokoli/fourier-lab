from __future__ import annotations

import numpy as np

from core.fft_utils import fft2_shift, ifft2_shift


def circular_mask(shape: tuple[int, int], radius: int, mode: str = "low") -> np.ndarray:
    """
    Pravi kružnu masku za filtriranje u frekvencijskom domenu.

    Parametri:
    shape -- dimenzije spektra
    radius -- poluprečnik kružne oblasti koja se zadržava ili uklanja
    mode -- "low" za niskopropusni, "high" za visokopropusni filter

    Povratna vrednost:
    np.ndarray -- maska sa vrednostima 0 i 1
    """
    h, w = shape
    cy, cx = h // 2, w // 2

    # Koordinate svih tačaka u matrici
    y, x = np.ogrid[:h, :w]

    # Rastojanje svake tačke od centra spektra
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Maska koja zadržava frekvencije unutar zadanog poluprečnika
    low_mask = (dist <= radius).astype(np.float32)

    if mode == "low":
        return low_mask
    if mode == "high":
        return 1.0 - low_mask

    raise ValueError("mode must be 'low' or 'high'")


def apply_low_pass(image_gray: np.ndarray, radius: int) -> np.ndarray:
    """
    Primena niskopropusnog filtera u frekvencijskom domenu.
    Zadržavaju se niske frekvencije, a visoke se uklanjaju.
    """
    spectrum = fft2_shift(image_gray)
    mask = circular_mask(spectrum.shape, radius, mode="low")
    filtered = spectrum * mask
    return ifft2_shift(filtered)


def apply_high_pass(image_gray: np.ndarray, radius: int) -> np.ndarray:
    """
    Primena visokopropusnog filtera u frekvencijskom domenu.
    Zadržavaju se visoke frekvencije, a niske se uklanjaju.
    """
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
    Uklanja izabrane frekvencijske komponente oko zadatih centara.

    Parametri:
    image_gray -- ulazna grayscale slika
    centers -- liste centara frekvencijskih oblasti koje se uklanjaju
    notch_radius -- poluprečnik oblasti koja se briše oko svakog centra

    Povratna vrednost:
    np.ndarray -- filtrirana slika u prostornom domenu
    """
    spectrum = fft2_shift(image_gray)
    h, w = spectrum.shape

    # Počinje se od maske koja zadržava sve frekvencije
    mask = np.ones((h, w), dtype=np.float32)

    yy, xx = np.ogrid[:h, :w]
    for cy, cx in centers:
        # Uklanja se mala kružna oblast oko izabranog centra
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask[dist <= notch_radius] = 0.0

    filtered = spectrum * mask
    return ifft2_shift(filtered)