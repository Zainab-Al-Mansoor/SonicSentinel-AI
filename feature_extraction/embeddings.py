"""
YAMNet – pretrained audio knowledge (step 2 of the feature-engineering plan).

YAMNet is Google's audio event classifier trained on AudioSet (≈2 million YouTube clips, 521 sound classes such
as Gunshot, Glass, Shatter, Siren, Screaming, Dog, Vehicle horn …). For every 2-second segment we take the MEAN of
YAMNet's 521 class scores over its 0.96-s frames and add them to our own features. The final classifier is still
our own model trained on our data; YAMNet only supplies extra, very informative inputs.

Needs:  pip install tensorflow   (tensorflow-hub is not required)
The model is downloaded once (≈15 MB) into python_models/yamnet/. Set SONIC_YAMNET_URL to a local SavedModel folder to work offline.
"""
import os
from pathlib import Path

import numpy as np

YAMNET_URL = os.environ.get("SONIC_YAMNET_URL", "https://tfhub.dev/google/yamnet/1")
YAMNET_SR = 16000
N_YAMNET = 521
_model = None
_error = None


def available() -> bool:
    """True if TensorFlow + TF-Hub can be imported and YAMNet loads."""
    try:
        _load()
        return True
    except Exception:
        return False


def unavailable_reason() -> str:
    return str(_error) if _error else ""


YAMNET_DIR = Path(__file__).resolve().parent.parent / "python_models" / "yamnet"   # local copy (downloaded once)
_DOWNLOADS = [
    "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed",
    "https://www.kaggle.com/api/v1/models/google/yamnet/tensorFlow2/yamnet/1/download",
]


def _download(dest: Path) -> Path:
    """Download the YAMNet SavedModel once (tar.gz, ~15 MB) and unpack it into `dest`."""
    import io
    import tarfile
    import urllib.request
    errors = []
    for url in _DOWNLOADS:
        try:
            print(f"Downloading YAMNet from {url.split('?')[0]} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "SonicSentinel"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            tmp = dest.with_name(dest.name + "_tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
                t.extractall(tmp)
            found = next(tmp.rglob("saved_model.pb")).parent
            found.rename(dest) if found != tmp else tmp.rename(dest)
            return dest
        except Exception as exc:          # try the next mirror
            errors.append(f"{url.split('?')[0]}: {exc}")
    raise RuntimeError("YAMNet download failed -> " + " | ".join(errors))


def _load():
    """Loads YAMNet with plain TensorFlow (tf.saved_model.load). tensorflow-hub is NOT needed, which avoids
    its tf-keras / pkg_resources version problems."""
    global _model, _error
    if _model is not None:
        return _model
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        if YAMNET_URL.startswith("http"):
            path = YAMNET_DIR
            if not (path / "saved_model.pb").exists():
                _download(path)
        else:
            path = Path(YAMNET_URL)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
        _model = tf.saved_model.load(str(path))
        return _model
    except Exception as exc:
        _error = exc
        raise


def yamnet_names() -> list[str]:
    return [f"yamnet{i}" for i in range(N_YAMNET)]


def yamnet_scores(segments: list[np.ndarray], sr: int) -> np.ndarray:
    """Mean YAMNet class scores (n_segments x 521) for pre-processed mono segments at `sr`."""
    import librosa
    import tensorflow as tf
    m = _load()
    out = np.zeros((len(segments), N_YAMNET), dtype=np.float32)
    for i, s in enumerate(segments):
        w = librosa.resample(np.asarray(s, np.float32), orig_sr=sr, target_sr=YAMNET_SR) if sr != YAMNET_SR else s
        w = np.clip(np.asarray(w, np.float32), -1.0, 1.0)
        if len(w) < YAMNET_SR:                      # YAMNet needs at least ~1 s
            w = np.pad(w, (0, YAMNET_SR - len(w)))
        scores, _emb, _spec = m(tf.constant(w))
        out[i] = np.asarray(scores).mean(axis=0)
    return out


if __name__ == "__main__":        # python -m feature_extraction.embeddings   -> quick self-check
    if available():
        s = yamnet_scores([np.zeros(44100, np.float32)], 44100)
        print(f"YAMNet OK – {s.shape[1]} scores per segment ({YAMNET_URL})")
    else:
        print(f"YAMNet NOT available: {unavailable_reason()}\nInstall: pip install tensorflow")
