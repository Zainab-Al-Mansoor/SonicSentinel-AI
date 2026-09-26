"""Waveform and Mel-spectrogram image generation (PNG) with matplotlib."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless – required inside Flask
import matplotlib.pyplot as plt
import librosa
import librosa.display

BG = "#281D35"      # "Aura" theme – matches the web UI cards
FG = "#B3A6C9"
INK = "#EAE6F2"


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#4A3D5E")


def save_waveform(y: np.ndarray, sr: int, path: str | Path, segments: list | None = None,
                  highlight: tuple | None = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.4), dpi=110)
    fig.patch.set_facecolor(BG)
    t = np.arange(len(y)) / sr
    step = max(1, len(y) // 20000)          # decimate for fast drawing
    ax.plot(t[::step], y[::step], color="#D4C4E8", linewidth=0.6)
    if segments:
        for (s, e) in segments:
            ax.axvline(s, color="#6E6187", linewidth=0.5, linestyle=":")
    if highlight:
        ax.axvspan(highlight[0], highlight[1], color="#E8A6D0", alpha=0.28)
    ax.set_xlim(0, max(t[-1] if len(t) else 1, 1e-3))
    ax.set_xlabel("Time (s)", color=FG, fontsize=8)
    ax.set_ylabel("Amplitude", color=FG, fontsize=8)
    _style(ax, "Waveform")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)


def save_spectrogram(y: np.ndarray, sr: int, path: str | Path, n_mels: int = 128) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    fig.patch.set_facecolor(BG)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma")
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.tick_params(colors=FG, labelsize=7)
    ax.set_xlabel("Time (s)", color=FG, fontsize=8)
    ax.set_ylabel("Hz (Mel)", color=FG, fontsize=8)
    _style(ax, "Mel spectrogram")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)
