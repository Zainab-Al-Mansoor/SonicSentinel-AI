"""
Audio file validation (format, size, duration, sample rate, channels,
integrity, presence of an audio signal). Returns a ValidationResult with
user-friendly error messages.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.settings import SUPPORTED_FORMATS, MAX_UPLOAD_MB, MIN_DURATION_S, MAX_DURATION_S
from .loader import load_audio, AudioDecodeError, AudioData


@dataclass
class ValidationResult:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    audio: AudioData | None = None


def check_extension(filename: str) -> bool:
    return Path(filename).suffix.lower().lstrip(".") in SUPPORTED_FORMATS


def validate_file(path: str | Path, filename: str | None = None) -> ValidationResult:
    path = Path(path)
    filename = filename or path.name
    res = ValidationResult(ok=False)

    if not check_extension(filename):
        res.errors.append(f"Unsupported format '{Path(filename).suffix}'. "
                          f"Allowed: {', '.join(f.upper() for f in SUPPORTED_FORMATS)}.")
        return res
    size = path.stat().st_size
    if size == 0:
        res.errors.append("The file is empty.")
        return res
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        res.errors.append(f"File is too large ({size/1e6:.1f} MB). Maximum is {MAX_UPLOAD_MB} MB.")
        return res

    try:
        audio = load_audio(path)
    except AudioDecodeError as exc:
        res.errors.append(f"File integrity check failed: {exc}")
        return res
    except Exception as exc:  # damaged header, truncated file, ...
        res.errors.append(f"The file appears to be damaged or is not valid audio ({type(exc).__name__}).")
        return res
    return validate_audio(audio, res)


def validate_audio(audio: AudioData, res: ValidationResult | None = None,
                   check_duration: bool = True) -> ValidationResult:
    res = res or ValidationResult(ok=False)
    res.audio = audio
    if audio.sample_rate < 8000:
        res.errors.append(f"Sampling rate {audio.sample_rate} Hz is too low (minimum 8000 Hz).")
    if audio.channels < 1 or audio.channels > 8:
        res.errors.append(f"Unsupported channel count: {audio.channels}.")
    if check_duration and audio.duration < MIN_DURATION_S:
        res.errors.append(f"Recording is too short ({audio.duration:.2f}s). Minimum is {MIN_DURATION_S}s.")
    if check_duration and audio.duration > MAX_DURATION_S:
        res.errors.append(f"Recording is too long ({audio.duration:.0f}s). Maximum is {MAX_DURATION_S}s.")

    y = audio.samples if audio.samples.ndim == 1 else audio.samples.mean(axis=1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-4:
        res.errors.append("No usable audio signal was found (the recording is silent).")
    if audio.sample_rate < 16000 and not res.errors:
        res.warnings.append("Low sampling rate – high-frequency detail (glass, alarms) may be lost.")
    res.ok = not res.errors
    return res
