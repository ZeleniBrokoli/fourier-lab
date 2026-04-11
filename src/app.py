from __future__ import annotations

import numpy as np
import streamlit as st

from core.fft_utils import (
    load_image,
    resize_max,
    to_grayscale,
    fft2_shift,
    spectrum_magnitude,
)
from core.filters import apply_low_pass, apply_high_pass, apply_notch_filter
from core.reconstruction import reconstruct_from_top_coefficients
from core.steganography import encode_text, decode_text, estimate_capacity_bytes


st.set_page_config(
    page_title="Fourier Lab",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clip_rgb(image: np.ndarray) -> np.ndarray:
    img = np.real(image).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    ch = np.real(channel).astype(np.float32)
    mn = float(np.min(ch))
    mx = float(np.max(ch))
    if np.isclose(mn, mx):
        return np.zeros_like(ch, dtype=np.uint8)
    ch = (ch - mn) / (mx - mn)
    return np.clip(ch * 255.0, 0, 255).astype(np.uint8)


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    return np.stack([normalize_channel(image[..., i]) for i in range(3)], axis=-1)


def apply_rgb_filter(image_rgb: np.ndarray, filter_fn, *args, **kwargs) -> np.ndarray:
    channels = []
    for i in range(3):
        ch = image_rgb[..., i].astype(np.float32)
        filtered = filter_fn(ch, *args, **kwargs)
        channels.append(filtered)
    return np.stack(channels, axis=-1)


def reconstruct_rgb(image_rgb: np.ndarray, keep_ratio: float) -> np.ndarray:
    channels = []
    for i in range(3):
        ch = image_rgb[..., i].astype(np.float32)
        recon = reconstruct_from_top_coefficients(ch, keep_ratio)
        channels.append(recon)
    return np.stack(channels, axis=-1)


def image_card(title: str, subtitle: str, accent: str = "primary") -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(238,242,255,0.85));
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 24px;
        padding: 18px 20px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 14px;
    ">
        <div style="font-size: 14px; color: #64748b; margin-bottom: 6px;">{subtitle}</div>
        <div style="font-size: 26px; font-weight: 800; color: #0f172a;">{title}</div>
    </div>
    """


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99,102,241,0.08), transparent 30%),
                radial-gradient(circle at top right, rgba(14,165,233,0.08), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eef2ff 0%, #ffffff 100%);
            border-right: 1px solid rgba(148,163,184,0.18);
        }
        .hero {
            background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(14,165,233,0.10));
            border: 1px solid rgba(99,102,241,0.12);
            border-radius: 28px;
            padding: 26px 28px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 16px;
        }
        .hero-title {
            font-size: 38px;
            font-weight: 900;
            color: #0f172a;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .hero-subtitle {
            font-size: 16px;
            color: #475569;
            max-width: 860px;
        }
        .section-title {
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
            margin: 4px 0 10px 0;
        }
        .hint {
            color: #64748b;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Fourier Lab</div>
            <div class="hero-subtitle">
                Interaktivna analiza slika u frekvencijskom domenu: pregled spektra, filtriranje,
                rekonstrukcija i steganografija — uz moderniji i čistiji interfejs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "Učitaj sliku",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    st.sidebar.markdown("### Kontrole")

    low_pass_radius = st.sidebar.slider("Low-pass radius", 5, 250, 40)
    high_pass_radius = st.sidebar.slider("High-pass radius", 5, 250, 40)
    keep_ratio = st.sidebar.slider("Rekonstrukcija — procenat frekvencija", 1, 100, 10) / 100.0

    st.sidebar.markdown("### Steganografija")
    stego_message = st.sidebar.text_area("Skrivena poruka", value="Pozdrav iz Fourier Lab-a!")
    inner_radius = st.sidebar.slider("Stego inner radius", 5, 120, 15)
    outer_radius = st.sidebar.slider("Stego outer radius", 20, 350, 200)

    if uploaded_file is None:
        st.info("Učitaj sliku iz sidebar-a da počneš.")
        return

    original = load_image(uploaded_file)
    original = resize_max(original, max_side=900)

    gray_for_fft = to_grayscale(original)
    spectrum = fft2_shift(gray_for_fft)
    spectrum_img = spectrum_magnitude(spectrum)

    h, w = original.shape[:2]

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(image_card("RGB", "kanali"), unsafe_allow_html=True)
    with col_b:
        st.markdown(image_card(f"{w} × {h}", "dimenzije"), unsafe_allow_html=True)
    with col_c:
        st.markdown(image_card("FFT", "analiza"), unsafe_allow_html=True)
    with col_d:
        st.markdown(image_card("Live", "interakcija"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Analiza", "Filteri i rekonstrukcija", "Steganografija"])

    with tab1:
        st.markdown('<div class="section-title">Original i frekvencijski spektar</div>', unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Originalna slika u boji")
            st.image(original, use_column_width=True)

        with right:
            st.subheader("FFT spektar (iz grayscale osnove)")
            st.image(spectrum_img, use_column_width=True)

        st.markdown(
            '<div class="hint">Spektar je računat nad grayscale verzijom slike, jer je to '
            'najčišći način za vizuelizaciju frekvencija. Original ostaje u boji.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown('<div class="section-title">Filtriranje u frekvencijskom domenu</div>', unsafe_allow_html=True)

        low_rgb = apply_rgb_filter(original, apply_low_pass, low_pass_radius)
        high_rgb = apply_rgb_filter(original, apply_high_pass, high_pass_radius)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("Original")
            st.image(original, use_column_width=True)

        with c2:
            st.write(f"Low-pass (radius = {low_pass_radius})")
            st.image(clip_rgb(low_rgb), use_column_width=True)

        with c3:
            st.write(f"High-pass (radius = {high_pass_radius})")
            st.image(normalize_rgb(high_rgb), use_column_width=True)

        st.divider()

        st.markdown('<div class="section-title">Rekonstrukcija iz najjačih frekvencija</div>', unsafe_allow_html=True)

        recon_rgb = reconstruct_rgb(original, keep_ratio)

        c4, c5 = st.columns(2)
        with c4:
            st.write(f"Koristi se {int(keep_ratio * 100)}% frekvencija")
            st.image(clip_rgb(recon_rgb), use_column_width=True)

        with c5:
            st.write("Original")
            st.image(original, use_column_width=True)

        st.caption("Što je manji keep ratio, slika je mutnija. Kako raste procenat, vraćaju se detalji.")

        st.divider()

        st.markdown('<div class="section-title">Notch filter demo</div>', unsafe_allow_html=True)
        st.caption("Ovo je primer uklanjanja odabranih frekvencija. Kasnije možemo dodati klik po spektru.")

        if st.checkbox("Pokaži primer notch filtera na fiksnim tačkama", value=False):
            h2, w2 = gray_for_fft.shape
            demo_centers = [
                (h2 // 2 + 30, w2 // 2 + 30),
                (h2 // 2 - 30, w2 // 2 - 30),
            ]
            notch_gray = apply_notch_filter(gray_for_fft, demo_centers, notch_radius=10)
            st.image(notch_gray, use_column_width=True)

    with tab3:
        st.markdown('<div class="section-title">Skrivena poruka u frekvencijskom domenu</div>', unsafe_allow_html=True)

        max_outer = max(20, min(350, min(h, w) // 2 - 2))
        safe_outer = min(outer_radius, max_outer)
        safe_inner = min(inner_radius, safe_outer - 1)

        capacity_bytes = estimate_capacity_bytes(
            gray_for_fft.shape,
            inner_radius=safe_inner,
            outer_radius=safe_outer,
        )
        st.write(f"Približan kapacitet: **{capacity_bytes} bajtova**")

        c6, c7 = st.columns(2)

        with c6:
            if st.button("Embed poruku"):
                try:
                    stego = encode_text(
                        gray_for_fft,
                        stego_message,
                        inner_radius=safe_inner,
                        outer_radius=safe_outer,
                    )
                    st.session_state["stego_image"] = stego
                    st.success("Poruka je ubačena u sliku.")
                    st.image(stego, use_column_width=True)
                except Exception as exc:
                    st.error(f"Greška pri embedovanju: {exc}")

        with c7:
            if st.button("Extract poruku"):
                try:
                    if "stego_image" in st.session_state:
                        source = st.session_state["stego_image"]
                    else:
                        source = gray_for_fft.astype("uint8")

                    decoded = decode_text(
                        source,
                        inner_radius=safe_inner,
                        outer_radius=safe_outer,
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