"""
Start SonicSentinel AI.

    python run.py                  # http://127.0.0.1:5000
    python run.py --host 0.0.0.0   # reachable from other devices on the LAN

Live microphone monitoring needs a "secure context": use http://localhost
or http://127.0.0.1 on the same computer, or serve over HTTPS.
"""
import argparse
import os

from src import create_app

app = create_app()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)
