from __future__ import annotations

import numpy as np
import streamlit as st

from core.fft_utils import normalize_image


def display_image(image: np.ndarray, caption: str | None = None) -> None:
    """
    Pomoćna funkcija za prikaz slike.

    Automatski normalizuje grayscale slike,
    dok RGB slike prikazuje direktno.
    """

    # Ako je slika grayscale (2D), potrebno je normalizovati vrednosti
    if image.ndim == 2:
        st.image(normalize_image(image), caption=caption, use_column_width=True)
    else:
        # RGB slika
        st.image(image.astype(np.uint8), caption=caption, use_column_width=True)


def show_image_block(title: str, image: np.ndarray, caption: str | None = None) -> None:
    """
    Prikazuje jednu sliku sa naslovom.
    """
    st.subheader(title)
    display_image(image, caption)


def show_side_by_side(
    left_title: str,
    left_image: np.ndarray,
    right_title: str,
    right_image: np.ndarray,
    left_caption: str | None = None,
    right_caption: str | None = None,
) -> None:
    """
    Prikazuje dve slike jednu pored druge.
    """

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(left_title)
        display_image(left_image, left_caption)

    with col2:
        st.subheader(right_title)
        display_image(right_image, right_caption)