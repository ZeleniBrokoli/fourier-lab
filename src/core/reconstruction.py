from __future__ import annotations

import numpy as np

from core.fft_utils import fft2_shift, ifft2_shift


def reconstruct_from_top_coefficients(
    image_gray: np.ndarray,
    keep_ratio: float,
) -> np.ndarray:
    """
    Rekonstruiše sliku koristeći samo najjače FFT koeficijente.

    Parametri:
    image_gray -- grayscale slika
    keep_ratio -- procenat frekvencija koje zadržavamo:
                  0.0 znači gotovo ništa,
                  1.0 znači potpuna rekonstrukcija

    Povratna vrednost:
    np.ndarray -- rekonstruisana slika
    """
    # Ograničavanje vrednosti na opseg [0, 1]
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))

    # Računanje Furijeove transformacije slike
    spectrum = fft2_shift(image_gray)

    # Magnituda svih koeficijenata pretvorena u jednodimenzioni niz
    flat_magnitude = np.abs(spectrum).ravel()

    # Broj koeficijenata koje zadržavamo
    total = flat_magnitude.size
    keep_count = max(1, int(total * keep_ratio))

    # Ako zadržavamo sve koeficijente, nema potrebe za filtriranjem
    if keep_count >= total:
        return ifft2_shift(spectrum)

    # Biramo indekse najjačih koeficijenata
    idx = np.argpartition(flat_magnitude, -keep_count)[-keep_count:]

    # Pravimo masku koja zadržava samo izabrane koeficijente
    mask = np.zeros(total, dtype=np.float32)
    mask[idx] = 1.0
    mask = mask.reshape(spectrum.shape)

    # Primena maske i povratak u prostorni domen
    filtered = spectrum * mask
    return ifft2_shift(filtered)


def progressive_reconstruction_frames(
    image_gray: np.ndarray,
    steps: int = 20,
) -> list[tuple[float, np.ndarray]]:
    """
    Generiše niz frejmova za postepenu rekonstrukciju slike.

    Parametri:
    image_gray -- grayscale slika
    steps -- broj koraka između minimalnog i maksimalnog zadržavanja frekvencija

    Povratna vrednost:
    list[tuple[float, np.ndarray]] -- lista parova
        (procena zadržanih frekvencija, rekonstruisana slika)
    """
    # Najmanje dva koraka da bi animacija imala smisla
    steps = max(2, int(steps))

    # Krećemo od malog procenta frekvencija pa do pune rekonstrukcije
    ratios = np.linspace(0.01, 1.0, steps)

    frames: list[tuple[float, np.ndarray]] = []

    for ratio in ratios:
        # Rekonstrukcija za svaki nivo zadržanih koeficijenata
        recon = reconstruct_from_top_coefficients(image_gray, float(ratio))
        frames.append((float(ratio), recon))

    return frames