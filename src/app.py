from __future__ import annotations

import streamlit as st

from core.fft_utils import (
    load_image,
    resize_max,
    to_grayscale,
    fft2_shift,
    spectrum_magnitude,
    normalize_image,
)
from core.filters import apply_low_pass, apply_high_pass, apply_notch_filter
from core.reconstruction import reconstruct_from_top_coefficients
from core.steganography import encode_text, decode_text, estimate_capacity_bytes
from ui.components import show_side_by_side


st.set_page_config(
    page_title="Fourier Lab",
    page_icon="✨",
    layout="wide",
)


def main() -> None:
    st.title("Fourier Lab")
    st.caption(
        "Interaktivna analiza, filtriranje, rekonstrukcija i steganografija u frekvencijskom domenu."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Učitaj sliku", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )

    st.sidebar.header("Podešavanja")
    low_pass_radius = st.sidebar.slider("Low-pass radius", 5, 250, 40)
    high_pass_radius = st.sidebar.slider("High-pass radius", 5, 250, 40)
    keep_ratio = st.sidebar.slider("Rekonstrukcija - procenat frekvencija", 1, 100, 10) / 100.0

    stego_message = st.sidebar.text_area("Skrivena poruka", value="Pozdrav iz Fourier Lab-a!")
    inner_radius = st.sidebar.slider("Stego inner radius", 5, 120, 15)
    outer_radius = st.sidebar.slider("Stego outer radius", 20, 350, 200)

    if uploaded_file is None:
        st.info("Učitaj sliku iz sidebar-a da počneš.")
        return

    original = load_image(uploaded_file)
    original = resize_max(original, max_side=900)
    gray = to_grayscale(original)

    spectrum = fft2_shift(gray)
    spectrum_img = spectrum_magnitude(spectrum)

    tab1, tab2, tab3 = st.tabs(["Analiza", "Filteri i rekonstrukcija", "Steganografija"])

    with tab1:
        show_side_by_side(
            "Original",
            original,
            "FFT spektar",
            spectrum_img,
        )

    with tab2:
        st.subheader("Filtriranje")

        low = apply_low_pass(gray, low_pass_radius)
        high = apply_high_pass(gray, high_pass_radius)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Original")
            st.image(normalize_image(gray), use_column_width=True)

        with col2:
            st.write(f"Low-pass (radius = {low_pass_radius})")
            st.image(normalize_image(low), use_column_width=True)

        with col3:
            st.write(f"High-pass (radius = {high_pass_radius})")
            st.image(normalize_image(high), use_column_width=True)

        st.divider()

        st.subheader("Rekonstrukcija iz najjačih frekvencija")
        recon = reconstruct_from_top_coefficients(gray, keep_ratio)

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Koristi se {int(keep_ratio * 100)}% frekvencija")
            st.image(normalize_image(recon), use_column_width=True)

        with c2:
            st.write("Original")
            st.image(normalize_image(gray), use_column_width=True)

        st.caption(
            "Manji procenat frekvencija daje mutniju sliku; kako raste keep ratio, vraćaju se detalji."
        )

        st.divider()

        st.subheader("Notch filter demo")
        st.caption("Ovaj deo je spreman za sledeći korak kada budemo dodali ručno biranje frekvencijskih tačaka.")
        if st.checkbox("Pokaži primer notch filtera na fiksnim tačkama", value=False):
            h, w = gray.shape
            demo_centers = [
                (h // 2 + 30, w // 2 + 30),
                (h // 2 - 30, w // 2 - 30),
            ]
            notch = apply_notch_filter(gray, demo_centers, notch_radius=10)
            st.image(normalize_image(notch), use_column_width=True)

    with tab3:
        st.subheader("Skrivena poruka u frekvencijskom domenu")
        capacity_bytes = estimate_capacity_bytes(gray.shape, inner_radius=inner_radius, outer_radius=outer_radius)
        st.write(f"Približan kapacitet: **{capacity_bytes} bajtova**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Embed poruku"):
                try:
                    stego = encode_text(
                        gray,
                        stego_message,
                        inner_radius=inner_radius,
                        outer_radius=outer_radius,
                    )
                    st.session_state["stego_image"] = stego
                    st.success("Poruka je ubačena u sliku.")
                    st.image(stego, caption="Stego slika", use_column_width=True)
                except Exception as exc:
                    st.error(f"Greška pri embedovanju: {exc}")

        with col2:
            if st.button("Extract poruku"):
                try:
                    if "stego_image" in st.session_state:
                        source = st.session_state["stego_image"]
                    else:
                        source = gray.astype("uint8")

                    decoded = decode_text(
                        source,
                        inner_radius=inner_radius,
                        outer_radius=outer_radius,
                    )
                    st.success("Poruka je izvučena.")
                    st.code(decoded)
                except Exception as exc:
                    st.error(f"Greška pri ekstrakciji: {exc}")

        if "stego_image" in st.session_state:
            st.divider()
            st.subheader("Stego slika")
            st.image(st.session_state["stego_image"], use_column_width=True)


if __name__ == "__main__":
    main()