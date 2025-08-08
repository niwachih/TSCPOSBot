from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello from Cloud Run", 200

@app.route("/health")
def health():
    return "OK", 200
