from __future__ import annotations

import numpy as np

from core.fft_utils import fft2_shift, ifft2_shift


HEADER_BYTES = 4  # 32-bit length header


def _bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bit length must be divisible by 8.")

    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | int(bit)
        out.append(byte)
    return bytes(out)


def _symmetric_coord(shape: tuple[int, int], x: int, y: int) -> tuple[int, int]:
    h, w = shape
    return (h - 1 - x, w - 1 - y)


def _embedding_coords(
    shape: tuple[int, int],
    count: int,
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> list[tuple[int, int]]:
    """
    Select coordinates in shifted FFT spectrum, avoiding the very center.
    Only one coordinate from each symmetric pair is returned.
    """
    h, w = shape
    cy, cx = h // 2, w // 2

    if outer_radius is None:
        outer_radius = min(cx, cy) - 2

    coords: list[tuple[int, int]] = []

    for x in range(h):
        for y in range(w):
            if x == cy and y == cx:
                continue

            # Keep only one from each symmetric pair
            if (x > cy) or (x == cy and y > cx):
                continue

            dist = np.sqrt((x - cy) ** 2 + (y - cx) ** 2)
            if inner_radius <= dist <= outer_radius:
                sx, sy = _symmetric_coord(shape, x, y)
                if 0 <= sx < h and 0 <= sy < w:
                    coords.append((x, y))

    coords.sort(key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)

    if len(coords) < count:
        raise ValueError(
            f"Not enough embedding capacity. Need {count} positions, got {len(coords)}."
        )

    return coords[:count]


def estimate_capacity_bytes(
    image_shape: tuple[int, int],
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> int:
    """Approximate payload capacity in bytes, including the 4-byte length header."""
    dummy_count = len(_embedding_coords(image_shape, 1, inner_radius, outer_radius))
    h, w = image_shape
    cy, cx = h // 2, w // 2

    if outer_radius is None:
        outer_radius = min(cx, cy) - 2

    available = 0
    for x in range(h):
        for y in range(w):
            if x == cy and y == cx:
                continue
            if (x > cy) or (x == cy and y > cx):
                continue
            dist = np.sqrt((x - cy) ** 2 + (y - cx) ** 2)
            if inner_radius <= dist <= outer_radius:
                available += 1

    # one bit per coordinate -> 8 coords per byte, minus 4 bytes header
    total_bytes = available // 8
    return max(0, total_bytes - HEADER_BYTES)


def encode_text(
    image_gray: np.ndarray,
    text: str,
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> np.ndarray:
    """
    Encode UTF-8 text into FFT coefficients.
    Returns a uint8 grayscale image.
    """
    image_gray = image_gray.astype(np.float32)
    spectrum = fft2_shift(image_gray)

    payload = text.encode("utf-8")
    header = len(payload).to_bytes(HEADER_BYTES, byteorder="big")
    bits = _bytes_to_bits(header + payload)

    coords = _embedding_coords(
        spectrum.shape,
        len(bits),
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )

    for bit, (x, y) in zip(bits, coords):
        sx, sy = _symmetric_coord(spectrum.shape, x, y)

        coeff = spectrum[x, y]
        magnitude = abs(coeff.real) + 30.0
        new_real = magnitude if bit == 1 else -magnitude

        spectrum[x, y] = new_real + 1j * coeff.imag
        spectrum[sx, sy] = np.conj(spectrum[x, y])

    stego = ifft2_shift(spectrum)
    stego = np.clip(np.real(stego), 0, 255).astype(np.uint8)
    return stego


def decode_text(
    image_gray: np.ndarray,
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> str:
    """
    Decode UTF-8 text from a stego image.
    """
    image_gray = image_gray.astype(np.float32)
    spectrum = fft2_shift(image_gray)

    max_possible_coords = len(
        _embedding_coords(spectrum.shape, 1, inner_radius=inner_radius, outer_radius=outer_radius)
    )

    coords = _embedding_coords(
        spectrum.shape,
        max_possible_coords,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )

    extracted_bits: list[int] = []
    for x, y in coords:
        bit = 1 if np.real(spectrum[x, y]) >= 0 else 0
        extracted_bits.append(bit)

    if len(extracted_bits) < HEADER_BYTES * 8:
        raise ValueError("Not enough embedded data to read header.")

    header_bits = extracted_bits[: HEADER_BYTES * 8]
    header_bytes = _bits_to_bytes(header_bits)
    msg_len = int.from_bytes(header_bytes, byteorder="big")

    total_needed_bits = (HEADER_BYTES + msg_len) * 8
    if total_needed_bits > len(extracted_bits):
        raise ValueError("Image does not contain the full hidden message.")

    message_bits = extracted_bits[HEADER_BYTES * 8 : total_needed_bits]
    message_bytes = _bits_to_bytes(message_bits)
    return message_bytes.decode("utf-8", errors="replace")