"""
Audio loading + metadata extraction.

WAV / FLAC / OGG are decoded with `soundfile`. MP3 / M4A (and anything
soundfile cannot read) are decoded through FFmpeg, which must be installed
and on the PATH (see README -> Installation).
"""
import io
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioDecodeError(Exception):
    """Raised when a file cannot be decoded as audio."""


@dataclass
class AudioData:
    samples: np.ndarray      # float32, shape (n,) mono or (n, channels)
    sample_rate: int
    channels: int
    bit_depth: int | None
    format: str
    file_size: int

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0

    def metadata(self, filename: str) -> dict:
        return {
            "filename": filename,
            "format": self.format,
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "file_size": self.file_size,
        }


_SUBTYPE_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_U8": 8, "PCM_S8": 8,
                 "FLOAT": 32, "DOUBLE": 64}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int, int]:
    if not ffmpeg_available():
        raise AudioDecodeError("FFmpeg is not installed, so this format cannot be decoded.")
    # Probe the original channel count / sample rate by letting ffmpeg keep them.
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "wav", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioDecodeError("the file is damaged or uses an unsupported audio encoding")
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
    return data, sr, data.shape[1]


def load_audio(path: str | Path) -> AudioData:
    """Decode an audio file, keeping its native sample rate and channels."""
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    size = path.stat().st_size
    bit_depth = None
    try:
        info = sf.info(str(path))
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        channels = data.shape[1]
        bit_depth = _SUBTYPE_BITS.get(info.subtype)
    except Exception:
        data, sr, channels = _decode_with_ffmpeg(path)

    if data.size == 0:
        raise AudioDecodeError("The file contains no audio frames.")
    if not np.all(np.isfinite(data)):
        raise AudioDecodeError("The file contains corrupted (non-finite) samples.")
    samples = data[:, 0] if channels == 1 else data
    return AudioData(samples=samples.astype(np.float32), sample_rate=int(sr),
                     channels=int(channels), bit_depth=bit_depth, format=ext, file_size=size)


def load_bytes(raw: bytes, suffix: str = ".wav") -> AudioData:
    """Decode audio held in memory (used for live microphone windows)."""
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        return load_audio(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
