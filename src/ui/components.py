from __future__ import annotations

import numpy as np
import streamlit as st

from core.fft_utils import normalize_image


def show_image_block(title: str, image: np.ndarray, caption: str | None = None) -> None:
    st.subheader(title)
    if image.ndim == 2:
        st.image(normalize_image(image), caption=caption, use_column_width=True)
    else:
        st.image(image.astype(np.uint8), caption=caption, use_column_width=True)


def show_side_by_side(
    left_title: str,
    left_image: np.ndarray,
    right_title: str,
    right_image: np.ndarray,
    left_caption: str | None = None,
    right_caption: str | None = None,
) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(left_title)
        if left_image.ndim == 2:
            st.image(normalize_image(left_image), caption=left_caption, use_column_width=True)
        else:
            st.image(left_image.astype(np.uint8), caption=left_caption, use_column_width=True)

    with col2:
        st.subheader(right_title)
        if right_image.ndim == 2:
            st.image(normalize_image(right_image), caption=right_caption, use_column_width=True)
        else:
            st.image(right_image.astype(np.uint8), caption=right_caption, use_column_width=True)