"""
Lab 02: Wavelets and STFT.
by Yevhen Ponomarov, CS-S125
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

ThresholdMode = Literal["soft", "hard"]


def haar_dwt1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-level 1D Haar DWT.

    For odd-length inputs, pad one sample (edge/reflect policy, document choice).

    Args:
        x: 1D numeric signal.

    Returns:
        (approx, detail): each length ~N/2.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("Input x must be 1D.")

    if x.size % 2 != 0:
        x = np.pad(x, (0, 1), mode="edge")

    evens = x[0::2]
    odds = x[1::2]
    # orthonormal Haar
    return (evens + odds) / np.sqrt(2), (evens - odds) / np.sqrt(2)


def haar_idwt1(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """
    Invert one-level 1D Haar DWT.

    Args:
        approx: Approximation coefficients.
        detail: Detail coefficients.

    Returns:
        Reconstructed signal.
    """
    approx = np.asarray(approx, dtype=np.float32)
    detail = np.asarray(detail, dtype=np.float32)

    if approx.shape != detail.shape:
        raise ValueError("Approx and detail must have same shape")

    out = np.empty(approx.size * 2, dtype=approx.dtype)
    out[0::2] = (approx + detail) / np.sqrt(2)
    out[1::2] = (approx - detail) / np.sqrt(2)
    return out


def haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute one-level 2D separable Haar DWT for grayscale images.

    Args:
        image: 2D grayscale image.

    Returns:
        LL, (LH, HL, HH).
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape

    # row transform
    if w % 2 != 0:
        img = np.pad(img, ((0, 0), (0, 1)), mode="edge")
    row_l = (img[:, 0::2] + img[:, 1::2]) / np.sqrt(2)
    row_h = (img[:, 0::2] - img[:, 1::2]) / np.sqrt(2)

    # col transform
    if h % 2 != 0:
        row_l = np.pad(row_l, ((0, 1), (0, 0)), mode="edge")
        row_h = np.pad(row_h, ((0, 1), (0, 0)), mode="edge")

    ll = (row_l[0::2, :] + row_l[1::2, :]) / np.sqrt(2)
    lh = (row_l[0::2, :] - row_l[1::2, :]) / np.sqrt(2)
    hl = (row_h[0::2, :] + row_h[1::2, :]) / np.sqrt(2)
    hh = (row_h[0::2, :] - row_h[1::2, :]) / np.sqrt(2)

    return ll, (lh, hl, hh)


def haar_idwt2(LL: np.ndarray, bands: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Invert one-level 2D Haar DWT.

    Args:
        LL: Low-low sub-band.
        bands: Tuple `(LH, HL, HH)`.

    Returns:
        Reconstructed image (crop policy for odd sizes should be documented).
    """
    lh, hl, hh = bands

    # inverse cols
    row_l = np.empty((LL.shape[0] * 2, LL.shape[1]), dtype=LL.dtype)
    row_l[0::2, :] = (LL + lh) / np.sqrt(2)
    row_l[1::2, :] = (LL - lh) / np.sqrt(2)

    row_h = np.empty((hl.shape[0] * 2, hl.shape[1]), dtype=hl.dtype)
    row_h[0::2, :] = (hl + hh) / np.sqrt(2)
    row_h[1::2, :] = (hl - hh) / np.sqrt(2)

    # inverse rows
    out = np.empty((row_l.shape[0], row_l.shape[1] * 2), dtype=row_l.dtype)
    out[:, 0::2] = (row_l + row_h) / np.sqrt(2)
    out[:, 1::2] = (row_l - row_h) / np.sqrt(2)
    return out


def wavelet_threshold(coeffs: Any, threshold: float, mode: ThresholdMode = "soft") -> Any:
    """
    Apply thresholding to coefficient arrays.

    Args:
        coeffs: Array or nested tuples/lists of arrays.
        threshold: Non-negative threshold value.
        mode: `"soft"` or `"hard"`.

    Returns:
        Thresholded coefficients with same structure.
    """
    if isinstance(coeffs, (list, tuple)):
        return type(coeffs)(wavelet_threshold(c, threshold, mode) for c in coeffs)

    arr = np.asarray(coeffs)
    if mode == "hard":
        return np.where(np.abs(arr) > threshold, arr, 0.0)
    elif mode == "soft":
        return np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def wavelet_denoise(image: np.ndarray, levels: int,
                    threshold: float, mode: ThresholdMode = "soft") -> np.ndarray:
    """
    Denoise image via multi-level Haar thresholding.

    Args:
        image: 2D grayscale image.
        levels: Number of decomposition levels.
        threshold: Coefficient threshold.
        mode: `"soft"` or `"hard"`.

    Returns:
        Denoised image with deterministic behavior.
    """
    image = np.asarray(image, dtype=np.float32)
    shapes = []
    coeffs_list = []
    curr = image

    # forward
    for _ in range(levels):
        shapes.append(curr.shape)
        curr, bands = haar_dwt2(curr)
        coeffs_list.append(bands)

    # threshold
    coeffs_thresh = wavelet_threshold(coeffs_list, threshold, mode)

    # backward
    for i in reversed(range(levels)):
        curr = haar_idwt2(curr, coeffs_thresh[i])
        h, w = shapes[i]
        curr = curr[:h, :w]

    return curr


def stft1(
    x: np.ndarray,
    fs_hz: float,
    frame_len: int,
    hop_len: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for 1D signal using SciPy.

    Returns:
        `(freqs_hz, times_s, Zxx)` where `Zxx` is complex.
    """
    noverlap = frame_len - hop_len
    f, t, Zxx = signal.stft(x, fs=fs_hz, window=window, nperseg=frame_len, noverlap=noverlap)
    return f, t, Zxx


def spectrogram_magnitude(Zxx: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Convert STFT matrix to magnitude spectrogram.

    Args:
        Zxx: Complex STFT matrix.
        log_scale: If True, return `log(1 + magnitude)`.

    Returns:
        Non-negative finite magnitude matrix.
    """
    mag = np.abs(Zxx)
    if log_scale:
        mag = np.log(1.0 + mag)
    return mag


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Min-max normalize an array to `[0,255]` for visualization."""
    arr = np.asarray(x, dtype=np.float32)
    mn, mx = np.min(arr), np.max(arr)
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    ret = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(ret, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    """
    Lab 02 demo (skeleton).

    Expected behavior after implementation:
    - wavelet denoising demo on image from `./imgs/`
    - LL/LH/HL/HH band visualization
    - STFT spectrogram demo on synthetic chirp signal
    - save outputs to `./out/lab02/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 02 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab02",
                        help="Output directory (relative to repo root)")
    args = parser.parse_args()


    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    img = cv2.imdecode(np.fromfile(imgs_dir / args.img, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    # wavelet demo
    try:
        rng = np.random.default_rng(0)
        noisy = img.astype(np.float32) + rng.normal(0.0, 20.0, size=img.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)
        den = wavelet_denoise(noisy, levels=2, threshold=20.0, mode="soft")

        ll, (lh, hl, hh) = haar_dwt2(img.astype(np.float32))

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Original", img),
                ("Noisy (Gaussian)", noisy),
                ("Wavelet denoised", den),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_denoise.png")

        plt.figure(figsize=(10, 8))
        for i, (title, band) in enumerate(
            [
                ("LL", ll),
                ("LH", lh),
                ("HL", hl),
                ("HH", hh),
            ],
            start=1,
        ):
            plt.subplot(2, 2, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(band), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_bands.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # STFT bridge demo
    try:
        fs = 400.0
        duration_s = 2.0
        t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
        f0, f1 = 15.0, 120.0
        k = (f1 - f0) / duration_s
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.sin(phase)

        freqs, times, zxx = stft1(x, fs_hz=fs, frame_len=128, hop_len=32, window="hann")
        mag = spectrogram_magnitude(zxx, log_scale=True)

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(times, freqs, mag, shading="gouraud")
        plt.title("STFT Spectrogram (log-magnitude)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="log(1 + |Zxx|)")
        save_figure(out_dir / "stft_spectrogram.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 02 demo is incomplete. Implement the TODO functions in labs/lab02_wavelets_stft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
