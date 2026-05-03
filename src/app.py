from __future__ import annotations

import numpy as np
import streamlit as st
import io
import time
import matplotlib.pyplot as plt
from PIL import Image

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


st.set_page_config(
    page_title="Fourier Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "stego_image" not in st.session_state:
    st.session_state["stego_image"] = None


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

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    mse_val = mse(a, b)
    if mse_val == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse_val))

def download_image(image: np.ndarray, filename: str):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    st.download_button(
        label="Download image",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )


def render_metric(title: str, value: str, subtitle: str) -> str:
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 22px;
        padding: 18px 20px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        min-height: 122px;
    ">
        <div style="font-size: 13px; color: #64748b; margin-bottom: 8px; font-weight: 600;">{subtitle}</div>
        <div style="font-size: 28px; font-weight: 800; color: #0f172a; line-height: 1.15;">{value}</div>
        <div style="font-size: 14px; color: #334155; margin-top: 6px;">{title}</div>
    </div>
    """


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f7fb;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div {
            color: #e5e7eb !important;
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #eef2ff 100%);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 26px;
            padding: 24px 26px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            font-size: 15px;
            color: #475569;
            max-width: 860px;
        }

        .section-title {
            font-size: 20px;
            font-weight: 800;
            color: #0f172a;
            margin: 6px 0 12px 0;
        }

        .soft-card {
            background: #ffffff;
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 22px;
            padding: 18px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        .hint {
            color: #64748b;
            font-size: 13px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.6rem 1rem;
            background: #ffffff;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
            color: white !important;
            border: none !important;
        }

        button[kind="secondary"], button[kind="primary"] {
            border-radius: 14px !important;
            padding: 0.6rem 1rem !important;
        }
        
        /* FILE UPLOADER - white clean fix */

        div[data-testid="stFileUploader"] {
            background: transparent !important;
        }

        /* unutrašnji beli box */
        div[data-testid="stFileUploader"] section {
            background: white !important;
            border-radius: 16px !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* GLAVNI TEKST */
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {
                 color: #0f172a !important;  /* TAMNO */
        }

        /* sekundarni tekst */
        div[data-testid="stFileUploader"] small {
            color: #475569 !important;
        }

        /* dugme */
        div[data-testid="stFileUploader"] button {
            background-color: #4f46e5 !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
        }

        /* TEXT AREA (steganografija) */
        
        div[data-testid="stTextArea"] textarea {
            background: white !important;
            color: #0f172a !important;   /* TAMAN TEKST */
            border-radius: 14px !important;
            border: 1px solid #e2e8f0 !important;
            padding: 10px !important;
        }
        
        /* placeholder tekst */
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #64748b !important;
        }

        /* label iznad */
        div[data-testid="stTextArea"] label {
            color: #e5e7eb !important;   /* da ostane vidljivo na tamnom sidebaru */
        }
        
        div[data-testid="stTextArea"] textarea:focus {
            border: 1px solid #4f46e5 !important;
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
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
                Interaktivna obrada slika kroz frekvencijski domen: analiza spektra, filtriranje,
                rekonstrukcija i steganografija.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "Učitaj sliku",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    st.sidebar.markdown("### Podešavanja")

    mode = st.sidebar.radio(
        "Način obrade",
        ["RGB (u boji)", "Grayscale (crno-belo)"],
    )

    low_pass_radius = st.sidebar.slider("Low-pass radius", 5, 250, 40)
    high_pass_radius = st.sidebar.slider("High-pass radius", 5, 250, 40)
    keep_ratio = st.sidebar.slider("Rekonstrukcija — procenat frekvencija", 1, 100, 10) / 100.0

    st.sidebar.markdown("### Steganografija")
    stego_message = st.sidebar.text_area("Skrivena poruka", value="Ubacimo skrivenu poruku.")
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

    use_color = mode == "RGB (u boji)"
    h, w = original.shape[:2]

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(render_metric("Slika je spremna", "RGB / BW", "Mode"), unsafe_allow_html=True)
    with col_b:
        st.markdown(render_metric(f"{w} × {h}", "Dimenzije", "Ulaz"), unsafe_allow_html=True)
    with col_c:
        st.markdown(render_metric("Frekvencijska analiza", "FFT", "Core"), unsafe_allow_html=True)
    with col_d:
        st.markdown(render_metric("Interaktivni alat", "Live", "UI"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Analiza", "Filteri i rekonstrukcija", "Steganografija"])

    with tab1:
        st.markdown('<div class="section-title">Original i frekvencijski spektar</div>', unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Originalna slika")
            st.image(original, use_column_width=True)

        with right:
            st.subheader("FFT spektar")
            st.image(spectrum_img, use_column_width=True)

        st.markdown(
            '<div class="hint">FFT spektar se računa nad grayscale verzijom slike radi standardne i pregledne '
            'vizuelizacije frekvencija, dok original ostaje u boji.</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### RGB kanali")

        r, g, b = original[..., 0], original[..., 1], original[..., 2]

        c1, c2, c3 = st.columns(3)
        c1.image(r, caption="Red", use_column_width=True)
        c2.image(g, caption="Green", use_column_width=True)
        c3.image(b, caption="Blue", use_column_width=True)

        st.info("Niske frekvencije = struktura, visoke frekvencije = ivice i detalji.")

    with tab2:
        st.markdown('<div class="section-title">Filtriranje u frekvencijskom domenu</div>', unsafe_allow_html=True)

        if use_color:
            low_img = apply_rgb_filter(original, apply_low_pass, low_pass_radius)
            high_img = apply_rgb_filter(original, apply_high_pass, high_pass_radius)
        else:
            low_img = apply_low_pass(gray_for_fft, low_pass_radius)
            high_img = apply_high_pass(gray_for_fft, high_pass_radius)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("Original")
            st.image(original, use_column_width=True)

        with c2:
            st.write(f"Low-pass (radius = {low_pass_radius})")
            if use_color:
                st.image(clip_rgb(low_img), use_column_width=True)
            else:
                st.image(normalize_image(low_img), use_column_width=True)

        with c3:
            st.write(f"High-pass (radius = {high_pass_radius})")
            if use_color:
                st.image(normalize_rgb(high_img), use_column_width=True)
            else:
                st.image(normalize_image(high_img), use_column_width=True)

        download_image(
            clip_rgb(low_img) if use_color else normalize_image(low_img),
            "low_pass.png"
        )

        download_image(
            normalize_rgb(high_img) if use_color else normalize_image(high_img),
            "high_pass.png"
        )

        st.divider()

        st.markdown('<div class="section-title">Rekonstrukcija iz najjačih frekvencija</div>', unsafe_allow_html=True)

        if use_color:
            recon_img = reconstruct_rgb(original, keep_ratio)
        else:
            recon_img = reconstruct_from_top_coefficients(gray_for_fft, keep_ratio)

        c4, c5 = st.columns(2)
        with c4:
            st.write(f"Koristi se {int(keep_ratio * 100)}% frekvencija")
            if use_color:
                st.image(clip_rgb(recon_img), use_column_width=True)
            else:
                st.image(normalize_image(recon_img), use_column_width=True)

        with c5:
            st.write("Original")
            st.image(original, use_column_width=True)

        st.caption("Što je manji keep ratio, slika je mutnija. Kako raste procenat, vraćaju se detalji.")

        st.markdown("### Kvalitet rekonstrukcije")

        if not use_color:
            st.write("MSE:", round(mse(gray_for_fft, recon_img), 2))
            st.write("PSNR:", round(psnr(gray_for_fft, recon_img), 2), "dB")
        else:
            # za RGB uzmi grayscale verziju za poređenje
            recon_gray = to_grayscale(recon_img)
            st.write("MSE:", round(mse(gray_for_fft, recon_gray), 2))
            st.write("PSNR:", round(psnr(gray_for_fft, recon_gray), 2), "dB")

        st.info("Manji MSE i veći PSNR znače bolju rekonstrukciju slike.")

        download_image(
            clip_rgb(recon_img) if use_color else normalize_image(recon_img),
            "reconstruction.png"
        )

        st.markdown("### Animacija rekonstrukcije")

        if st.button("Pokreni animaciju"):
            placeholder = st.empty()

            for i in range(5, 100, 5):
                if use_color:
                    img = reconstruct_rgb(original, i / 100)
                    display = clip_rgb(img)
                else:
                    img = reconstruct_from_top_coefficients(gray_for_fft, i / 100)
                    display = normalize_image(img)

                placeholder.image(display, use_column_width=True)
                time.sleep(0.1)

        st.divider()

        st.markdown('<div class="section-title">Notch filter demo</div>', unsafe_allow_html=True)
        st.caption("Primer uklanjanja određenih frekvencija.")

        if st.checkbox("Pokaži primer notch filtera na fiksnim tačkama", value=False):
            h2, w2 = gray_for_fft.shape
            demo_centers = [
                (h2 // 2 + 30, w2 // 2 + 30),
                (h2 // 2 - 30, w2 // 2 - 30),
            ]
            notch_gray = apply_notch_filter(gray_for_fft, demo_centers, notch_radius=10)
            st.image(normalize_image(notch_gray), use_column_width=True)

        st.divider()
        st.markdown("### Histogram")

        fig, ax = plt.subplots()
        ax.hist(gray_for_fft.flatten(), bins=50)
        st.pyplot(fig)

    with tab3:
        st.markdown('<div class="section-title">Skrivena poruka u frekvencijskom domenu</div>', unsafe_allow_html=True)

        max_outer = max(20, min(350, min(h, w) // 2 - 2))
        safe_outer = min(outer_radius, max_outer)
        safe_inner = min(inner_radius, max(5, safe_outer - 1))

        capacity_bytes = estimate_capacity_bytes(
            gray_for_fft.shape,
            inner_radius=safe_inner,
            outer_radius=safe_outer,
        )
        st.write(f"Približan kapacitet: **{capacity_bytes} bajtova**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original (grayscale)")
            st.image(normalize_image(gray_for_fft), use_column_width=True)

        with col2:
            st.subheader("Stego slika")
            if st.session_state["stego_image"] is not None:
                st.image(st.session_state["stego_image"], use_column_width=True)
            else:
                st.info("Još nije napravljena stego slika.")

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Embed poruku", key="embed_button"):
                try:
                    stego = encode_text(
                        gray_for_fft,
                        stego_message,
                        inner_radius=safe_inner,
                        outer_radius=safe_outer,
                    )
                    st.session_state["stego_image"] = stego
                    st.success("Poruka je ubačena u sliku.")
                except Exception as exc:
                    st.error(f"Greška pri embedovanju: {exc}")

        with c2:
            if st.button("Extract poruku", key="extract_button"):
                try:
                    source = st.session_state["stego_image"]
                    if source is None:
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

        if st.session_state["stego_image"] is not None:
            st.divider()
            st.subheader("Poređenje")
            left, right = st.columns(2)
            with left:
                st.write("Original")
                st.image(normalize_image(gray_for_fft), use_column_width=True)
            with right:
                st.write("Stego")
                st.image(st.session_state["stego_image"], use_column_width=True)

if __name__ == "__main__":
    main()
