import os
from pathlib import Path

import numpy as np

YAMNET_URL = os.environ.get("SONIC_YAMNET_URL", "https://tfhub.dev/google/yamnet/1")
YAMNET_SR = 16000
N_YAMNET = 521
_model = None
_error = None


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def unavailable_reason() -> str:
    return str(_error) if _error else ""


YAMNET_DIR = Path(__file__).resolve().parent.parent / "python_models" / "yamnet"
_DOWNLOADS = [
    "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed",
    "https://www.kaggle.com/api/v1/models/google/yamnet/tensorFlow2/yamnet/1/download",
]


def _download(dest: Path) -> Path:
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
        except Exception as exc:
            errors.append(f"{url.split('?')[0]}: {exc}")
    raise RuntimeError("YAMNet download failed -> " + " | ".join(errors))


def _load():
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
    import librosa
    import tensorflow as tf
    m = _load()
    out = np.zeros((len(segments), N_YAMNET), dtype=np.float32)
    for i, s in enumerate(segments):
        w = librosa.resample(np.asarray(s, np.float32), orig_sr=sr, target_sr=YAMNET_SR) if sr != YAMNET_SR else s
        w = np.clip(np.asarray(w, np.float32), -1.0, 1.0)
        if len(w) < YAMNET_SR:
            w = np.pad(w, (0, YAMNET_SR - len(w)))
        scores, _emb, _spec = m(tf.constant(w))
        out[i] = np.asarray(scores).mean(axis=0)
    return out


if __name__ == "__main__":
    if available():
        s = yamnet_scores([np.zeros(44100, np.float32)], 44100)
        print(f"YAMNet OK – {s.shape[1]} scores per segment ({YAMNET_URL})")
    else:
        print(f"YAMNet NOT available: {unavailable_reason()}\nInstall: pip install tensorflow")
