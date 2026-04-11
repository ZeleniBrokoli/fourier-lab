from __future__ import annotations

import numpy as np

from core.fft_utils import fft2_shift, ifft2_shift

HEADER_BYTES = 4          # 32-bitna dužina poruke
REPEAT_FACTOR = 5         # svaki bit se upisuje 5 puta radi stabilnosti


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
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | int(bit)
        out.append(byte)
    return bytes(out)


def _majority_vote(group: list[int]) -> int:
    return 1 if sum(group) >= (len(group) / 2) else 0


def _symmetric_coord(shape: tuple[int, int], x: int, y: int) -> tuple[int, int]:
    h, w = shape
    return (h - 1 - x, w - 1 - y)


def _candidate_coords(
    shape: tuple[int, int],
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    cy, cx = h // 2, w // 2

    if outer_radius is None:
        outer_radius = min(cx, cy) - 2

    coords: list[tuple[int, int]] = []

    for x in range(h):
        for y in range(w):
            if x == cy and y == cx:
                continue

            # uzimamo samo jednu polovinu simetričnih parova
            if (x > cy) or (x == cy and y > cx):
                continue

            dist = np.sqrt((x - cy) ** 2 + (y - cx) ** 2)
            if inner_radius <= dist <= outer_radius:
                coords.append((x, y))

    # krećemo od stabilnijih frekvencija bližih sredini
    coords.sort(key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)
    return coords


def estimate_capacity_bytes(
    image_shape: tuple[int, int],
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> int:
    coords = _candidate_coords(image_shape, inner_radius, outer_radius)
    usable_bits = len(coords) // REPEAT_FACTOR
    total_bytes = usable_bits // 8
    return max(0, total_bytes - HEADER_BYTES)


def encode_text(
    image_gray: np.ndarray,
    text: str,
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> np.ndarray:
    image_gray = image_gray.astype(np.float32)
    spectrum = fft2_shift(image_gray)

    payload = text.encode("utf-8")
    header = len(payload).to_bytes(HEADER_BYTES, byteorder="big")
    bits = _bytes_to_bits(header + payload)

    repeated_bits: list[int] = []
    for bit in bits:
        repeated_bits.extend([bit] * REPEAT_FACTOR)

    coords = _candidate_coords(
        spectrum.shape,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )

    if len(repeated_bits) > len(coords):
        raise ValueError("Poruka je predugačka za ovu sliku.")

    for bit, (x, y) in zip(repeated_bits, coords):
        sx, sy = _symmetric_coord(spectrum.shape, x, y)

        coeff = spectrum[x, y]

        # jače i stabilnije upisivanje bita kroz znak realnog dela
        base = max(abs(coeff.real), abs(coeff.imag), 60.0)
        new_real = base if bit == 1 else -base
        new_coeff = new_real + 1j * coeff.imag

        spectrum[x, y] = new_coeff
        spectrum[sx, sy] = np.conj(new_coeff)

    stego = ifft2_shift(spectrum)
    stego = np.clip(np.real(stego), 0, 255).astype(np.uint8)
    return stego


def decode_text(
    image_gray: np.ndarray,
    inner_radius: int = 15,
    outer_radius: int | None = None,
) -> str:
    image_gray = image_gray.astype(np.float32)
    spectrum = fft2_shift(image_gray)

    coords = _candidate_coords(
        spectrum.shape,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )

    if len(coords) < REPEAT_FACTOR * HEADER_BYTES * 8:
        raise ValueError("Nema dovoljno podataka za čitanje zaglavlja.")

    raw_bits: list[int] = []
    for x, y in coords:
        raw_bits.append(1 if np.real(spectrum[x, y]) >= 0 else 0)

    # grupisanje i majority vote
    grouped_bits: list[int] = []
    usable_len = (len(raw_bits) // REPEAT_FACTOR) * REPEAT_FACTOR
    raw_bits = raw_bits[:usable_len]

    for i in range(0, len(raw_bits), REPEAT_FACTOR):
        grouped_bits.append(_majority_vote(raw_bits[i:i + REPEAT_FACTOR]))

    if len(grouped_bits) < HEADER_BYTES * 8:
        raise ValueError("Nema dovoljno podataka za čitanje zaglavlja.")

    header_bits = grouped_bits[: HEADER_BYTES * 8]
    header_bytes = _bits_to_bytes(header_bits)
    msg_len = int.from_bytes(header_bytes, byteorder="big")

    total_bits_needed = (HEADER_BYTES + msg_len) * 8
    if total_bits_needed > len(grouped_bits):
        raise ValueError("Slika ne sadrži celu skrivenu poruku.")

    message_bits = grouped_bits[HEADER_BYTES * 8 : total_bits_needed]
    message_bytes = _bits_to_bytes(message_bits)

    return message_bytes.decode("utf-8", errors="strict")