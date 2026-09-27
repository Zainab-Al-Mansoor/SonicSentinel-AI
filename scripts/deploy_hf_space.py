"""
Deploy SonicSentinel AI to a free Hugging Face Space (Docker) in one command.

    pip install huggingface_hub
    python scripts/deploy_hf_space.py --space YOUR-HF-USERNAME/sonicsentinel-ai

What it does
  1. checks that both trained models are present
  2. writes requirements-deploy.txt with the EXACT library versions installed on this PC
     (a joblib model must be loaded with the same scikit-learn / XGBoost / NumPy it was trained with)
  3. logs in to Hugging Face (asks for a token the first time)
  4. creates the Space (Docker SDK) if it does not exist and stores a random SONIC_SECRET_KEY as a secret
  5. uploads the project (datasets, database, uploads and other large/private folders are skipped)
     together with the Space README header (sdk: docker, app_port: 7860)

Run it again after any change to update the live app. Hugging Face then rebuilds the image (≈ 10–20 min).
"""
import argparse
import re
import secrets
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    ".git/*", ".venv/*", "venv/*", "**/__pycache__/*", "*.pyc", ".pytest_cache/*",
    "downloads/*", "audio_dataset/*", "gtm_model/training_samples/*", "_to_delete/*", "_training_pack/*",
    "data/*.db", "data/uploads/*", "data/segments/*", "data/images/*", "data/features/*", "data/exports/*",
    "data/reports/*", "logs/*", "notebooks/*", "screenshots/*", "python_models/yamnet/*",
    "python_models/yamnet_tmp/*", "reports/label_audit.html", "documentation/*.docx", "documentation/*.pdf",
    "README.md", "*.log", "Claude outputs/*",
]

SPACE_HEADER = """---
title: SonicSentinel AI
emoji: 🔊
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Acoustic safety monitoring with two AI models
---

"""


def pinned_requirements() -> str:
    """requirements.txt with every installed package pinned to the version on this PC."""
    out = ["# written by scripts/deploy_hf_space.py – exact versions of the training PC"]
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name = re.split(r"[<>=!~ ;\[]", s, maxsplit=1)[0]
        marker = s.split(";", 1)[1].strip() if ";" in s else ""
        if name.lower() in ("waitress", "gunicorn", "pytest"):
            continue                       # server installed separately; tests not needed in the image
        try:
            ver = metadata.version(name)
            out.append(f"{name}=={ver}" + (f" ; {marker}" if marker else ""))
        except metadata.PackageNotFoundError:
            out.append(s)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="YOUR-HF-USERNAME/space-name, e.g. zainab/sonicsentinel-ai")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    a = ap.parse_args()

    missing = [p for p in ("python_models/saved/sonic_model.joblib", "gtm_model/model/model.json",
                           "gtm_model/model/metadata.json", "gtm_model/model/weights.bin") if not (ROOT / p).exists()]
    if missing:
        sys.exit("Missing model files: " + ", ".join(missing))

    req = pinned_requirements()
    (ROOT / "requirements-deploy.txt").write_text(req, encoding="utf-8")
    print("requirements-deploy.txt written:\n  " + "\n  ".join(req.splitlines()[1:]))

    try:
        from huggingface_hub import HfApi, login, get_token
    except ImportError:
        sys.exit("Install the Hugging Face client first:  pip install huggingface_hub")
    if not get_token():
        print("\nPaste a Hugging Face WRITE token (huggingface.co → Settings → Access Tokens → New token → Write):")
        login()
    api = HfApi()

    print(f"\nCreating / updating Space {a.space} ...")
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    try:
        api.add_space_secret(a.space, "SONIC_SECRET_KEY", secrets.token_hex(32))
    except Exception as exc:
        print(f"[warn] could not set SONIC_SECRET_KEY automatically ({exc}); add it in the Space settings.")

    print("Uploading project files (large binaries go through Hugging Face storage automatically) ...")
    api.upload_folder(folder_path=str(ROOT), repo_id=a.space, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy SonicSentinel AI")
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=a.space,
                    repo_type="space", commit_message="Space README")

    user, name = a.space.split("/", 1)
    print("\nDone. Hugging Face is now building the Docker image (10–20 minutes the first time).")
    print(f"  Build logs : https://huggingface.co/spaces/{a.space}  (tab 'Logs')")
    print(f"  App (use this link – login and microphone work here): https://{user}-{name}.hf.space".lower())


if __name__ == "__main__":
    main()
