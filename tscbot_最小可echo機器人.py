import os
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# 讀環境變數
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_BOT_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_BOT_CHANNEL_SECRET")

line_bot_api = None
handler = None

# 安全初始化（沒有就先讓服務能跑起來，以免 503）
if CHANNEL_ACCESS_TOKEN and CHANNEL_SECRET:
    try:
        line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
        handler = WebhookHandler(CHANNEL_SECRET)
    except Exception as e:
        print(f"[WARN] LINE init failed: {e}")
else:
    print("[WARN] Missing LINE envs: LINE_BOT_CHANNEL_ACCESS_TOKEN / LINE_BOT_CHANNEL_SECRET")

@app.route("/")
def index():
    return "Hello from Cloud Run"

@app.route("/health")
def health():
    return "OK"

@app.route("/callback", methods=["POST"])
def callback():
    if handler is None:
        # 沒設定金鑰就回 200，避免 LINE 驗證失敗；但功能不會生效
        return "LINE handler not initialized", 200

    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400

    return "OK"

# 收到文字就回：1) 我收到了你的訊息 2) 原封不動 echo
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    if line_bot_api is None:
        return
    user_text = event.message.text
    msgs = [
        TextSendMessage(text="我收到了你的訊息"),
        TextSendMessage(text=user_text),
    ]
    line_bot_api.reply_message(event.reply_token, msgs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

