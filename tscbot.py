# tscbot.py — minimal health/webhook stub for Cloud Run
import os
from flask import Flask, request

app = Flask(__name__)

@app.get("/")
def root():
    return "OK", 200

@app.get("/health")
def health():
    return "OK", 200

@app.post("/callback")
def callback():
    # just acknowledge quickly for LINE webhook validation
    # you can inspect headers/body if you want to debug:
    # print("X-Line-Signature:", request.headers.get("X-Line-Signature"))
    # print("Body:", request.get_data(as_text=True))
    return "OK", 200

# Local run fallback (not used by gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
