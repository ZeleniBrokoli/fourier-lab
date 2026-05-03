from __future__ import annotations

import numpy as np
from PIL import Image


def load_image(uploaded_file) -> np.ndarray:
    """
    Učitava sliku i konvertuje je u RGB NumPy niz.

    Parametri:
    uploaded_file -- fajl koji korisnik učitava (npr. preko Streamlit-a)

    Povratna vrednost:
    np.ndarray -- slika u RGB formatu (uint8)
    """

    # Otvaranje slike i konverzija u RGB (bez obzira na originalni format)
    image = Image.open(uploaded_file).convert("RGB")

    # Pretvaranje u NumPy niz radi dalje obrade
    return np.array(image, dtype=np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    #Konvertuje RGB ili RGBA sliku u grayscale (sive tonove)
    if image.ndim == 2:
        return image.astype(np.float32)

    rgb = image[..., :3].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def resize_max(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    """
    Skalira sliku tako da njena duža stranica ne prelazi zadatu vrednost,
    uz očuvanje proporcija.

    Parametri:
    image -- ulazna slika
    max_side -- maksimalna dužina veće dimenzije slike

    Povratna vrednost:
    np.ndarray -- skalirana slika
    """

    # Dimenzije slike
    h, w = image.shape[:2]

    # Duža stranica
    longest = max(h, w)

    # Ako je slika već dovoljno mala, ne skalira se
    if longest <= max_side:
        return image

    # Faktor skaliranja
    scale = max_side / longest

    # Nove dimenzije (očuvanje odnosa stranica)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Konverzija u PIL radi lakšeg skaliranja
    pil_image = Image.fromarray(image.astype(np.uint8))

    # Promena veličine slike
    resized = pil_image.resize((new_w, new_h))

    # Povratak u NumPy format
    return np.array(resized, dtype=np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizuje sliku tako da njene vrednosti budu u opsegu [0, 255],
    što je pogodno za prikaz.

    Parametri:
    image -- ulazna slika (može imati proizvoljne numeričke vrednosti)

    Povratna vrednost:
    np.ndarray -- normalizovana slika tipa uint8
    """

    # Uzimamo realni deo (bitno jer FFT daje kompleksne vrednosti)
    img = np.real(image).astype(np.float32)

    # Minimalna i maksimalna vrednost u slici
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    # Ako su sve vrednosti iste, vraćamo crnu sliku (da izbegnemo deljenje nulom)
    if np.isclose(min_val, max_val):
        return np.zeros_like(img, dtype=np.uint8)

    # Linearna normalizacija na opseg [0, 1]
    img = (img - min_val) / (max_val - min_val)

    # Skaliranje na opseg [0, 255] i ograničavanje vrednosti
    img = np.clip(img * 255.0, 0, 255)

    return img.astype(np.uint8)


def fft2_shift(image_gray: np.ndarray) -> np.ndarray:
    """
    Računa 2D Furijeovu transformaciju slike i pomera nultu frekvenciju u centar.

    Parametri:
    image_gray -- ulazna slika u grayscale formatu

    Povratna vrednost:
    np.ndarray -- spektar slike u frekvencijskom domenu (kompleksne vrednosti)
    """

    # Pretvaranje u float radi numeričke stabilnosti
    image_gray = image_gray.astype(np.float32)

    # 2D FFT + pomeranje niske frekvencije u centar (radi vizualizacije)
    return np.fft.fftshift(np.fft.fft2(image_gray))


def ifft2_shift(spectrum_shifted: np.ndarray) -> np.ndarray:
    """
    Računa inverznu 2D Furijeovu transformaciju.

    Parametri:
    spectrum_shifted -- spektar sa centriranim frekvencijama

    Povratna vrednost:
    np.ndarray -- rekonstruisana slika (realne vrednosti)
    """

    # Vraćanje spektra u originalni raspored (obrnut fftshift)
    image = np.fft.ifft2(np.fft.ifftshift(spectrum_shifted))

    # Uzimamo realni deo (zbog numeričkih grešaka)
    return np.real(image)


def spectrum_magnitude(spectrum_shifted: np.ndarray) -> np.ndarray:
    """
    Računa logaritamski spektar magnitude Furijeove transformacije
    i normalizuje ga za prikaz.

    Parametri:
    spectrum_shifted -- spektar dobijen nakon FFT i fftshift operacije

    Povratna vrednost:
    np.ndarray -- slika spektra pogodna za prikaz
    """

    # Magnituda(jacina frekvencije) kompleksnog spektra
    magnitude = np.abs(spectrum_shifted)

    # Logaritamsko skaliranje radi bolje vizualizacije (smanjuje opseg vrednosti)
    magnitude = np.log1p(magnitude)

    # Normalizacija na opseg [0, 255] za prikaz
    return normalize_image(magnitude)


def split_rgb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Razdvaja RGB sliku na tri kanala (R, G, B).

    Parametri:
    image -- ulazna RGB slika

    Povratna vrednost:
    tuple -- (R, G, B) kanali kao zasebne matrice
    """

    # Uzimamo samo prva tri kanala (zanemarujemo alpha ako postoji)
    rgb = image[..., :3]

    return rgb[..., 0], rgb[..., 1], rgb[..., 2]