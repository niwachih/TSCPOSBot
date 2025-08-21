import os
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# Google Sheet安全初始化
import pygsheets
from google.oauth2 import service_account

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

# Google Sheet安全初始化
def init_google_sheet():
    """
    安全初始化 Google Sheets 連線
    - 檢查 service account key 是否存在
    - 檢查環境變數是否設定
    - 捕捉初始化錯誤，避免程式崩潰
    """
    service_account_file = 'service_account_key.json'
    sheet_url = os.environ.get("GOOGLESHEET_URL")

    # 檢查檔案是否存在
    if not os.path.exists(service_account_file):
        print(f"[警告] 找不到 {service_account_file}，無法初始化 Google Sheets。")
        return None

    # 檢查環境變數
    if not sheet_url:
        print("[警告] 環境變數 GOOGLESHEET_URL 未設定，無法初始化 Google Sheets。")
        return None

    try:
        # 授權
        gc = pygsheets.authorize(service_account_file=service_account_file)

        # 嘗試開啟試算表，加上 timeout 保護
        sheet = gc.open_by_url(sheet_url)
        print("[成功] 已連線至 Google Sheets。")
        return sheet

    except Exception as e:
        print(f"[錯誤] 初始化 Google Sheets 失敗：{e}")
        return None

sheet = init_google_sheet()
if sheet:
    # 可以安全使用 sheet
    wks = sheet.sheet1
    print("工作表名稱:", wks.title)
else:
    print("Google Sheets 尚未初始化，略過相關功能。")

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
# 僅在 handler 初始化成功後才註冊事件處理器，避免 None 導致匯入失敗
if handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        if line_bot_api is None:
            return
        user_text = event.message.text
        msgs = [
            TextSendMessage(text="我收到了你的訊息"),
            TextSendMessage(text=user_text),
        ]
        try:
            line_bot_api.reply_message(event.reply_token, msgs)
        except Exception as e:
            print(f"[WARN] reply_message failed: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
