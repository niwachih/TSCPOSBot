import os
import json
import time
import threading
from datetime import datetime

# Flask and LINE Bot imports
from flask import Flask, abort, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import *

# Google services imports
import pygsheets
import google.generativeai as genai
from google.cloud import firestore
from google.oauth2 import service_account

# ML and NLP imports
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

# Time zone
import pytz

###############################################################################
# CONFIGURATION AND INITIALIZATION
###############################################################################

# 設定版本代碼和時區
VERSION_CODE = "09.06.2025"
GMT_8 = pytz.timezone("Asia/Taipei")

print(f"Starting application - Version Code: {VERSION_CODE}")

app = Flask(__name__)

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
generation_model = genai.GenerativeModel("gemini-2.0-flash")

# LINE Bot setup
line_bot_api = LineBotApi(os.environ.get("LINE_BOT_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_BOT_CHANNEL_SECRET"))
ALLOWED_DESTINATION = os.environ.get("ALLOWED_DESTINATION")

# Google Sheets setup
gc = pygsheets.authorize(service_account_file='service_account_key.json')
sheet = gc.open_by_url(os.environ.get("GOOGLESHEET_URL"))

# Firestore setup
def get_firestore_client_from_env():
    firestore_json = os.getenv("FIRESTORE")
    if not firestore_json:
        raise ValueError("FIRESTORE environment variable is not set.")
    
    cred_info = json.loads(firestore_json)
    credentials = service_account.Credentials.from_service_account_info(cred_info)
    return firestore.Client(credentials=credentials, project=cred_info["project_id"])

db = get_firestore_client_from_env()

###############################################################################
# DATA LOADING AND PREPROCESSING
###############################################################################

# Load questions and answers from Google Sheets 主要QA
def load_sheet_data():
    # Main questions
    main_ws = sheet.worksheet("title", "表單回應")
    main_questions = main_ws.get_col(3, include_tailing_empty=False)
    main_answers = main_ws.get_col(4, include_tailing_empty=False)
    
    # 取得 "CPC問題" 和 "CPC點數" 的值
    cpc_ws = sheet.worksheet("title", "中油點數")
    cpc_questions = cpc_ws.get_col(8, include_tailing_empty=False)
    cpc_answers = cpc_ws.get_col(9, include_tailing_empty=False)
    cpc_list = cpc_ws.get_col(1, include_tailing_empty=False)
    
    return main_questions + cpc_questions, main_answers + cpc_answers, cpc_list

questions_in_sheet, answers_in_sheet, cpc_list = load_sheet_data()

# Load synonyms dictionary
def load_synonyms():
    syn_ws = sheet.worksheet("title", "同義詞")
    synonym_rows = syn_ws.get_all_values()
    
    synonym_dict = {}
    for row in synonym_rows:
        synonyms = [word.strip() for word in row if word.strip()]
        for word in synonyms:
            synonym_dict[word] = set(synonyms) - {word}
    
    return synonym_dict

synonym_dict = load_synonyms()

# Initialize ML models
# 先對問句進行分詞
tokenized_questions = [list(jieba.cut(q)) for q in questions_in_sheet]
# 建立 BM25 模型
bm25 = BM25Okapi(tokenized_questions)

# 載入中文句向量模型
_model = None
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

question_embeddings = get_model().encode(questions_in_sheet)

###############################################################################
# SEARCH AND RETRIEVAL FUNCTIONS
###############################################################################

def expand_query(query):
    """擴展查詢詞，加入同義詞"""
    words = jieba.lcut(query)
    expanded_words = set(words)
    
    for word in words:
        if word in synonym_dict:
            expanded_words.update(synonym_dict[word])
    
    return " ".join(expanded_words)

def retrieve_top_n(query, n=2, threshold=5, high_threshold=10):
    """取得最相似的問題
    ##作法
    1.使用Sentence Transformers進行相似度計算
    2.使用BM25強化搜索
    3.閥值為5，超過才列為答案
    4.最多選擇2個答案 
    """
    try:
        expanded_query = expand_query(query)
        tokenized_query = list(jieba.cut(expanded_query))
        # BM25 排序
        bm25_scores = bm25.get_scores(tokenized_query)
        # Sentence Transformers 相似度計算(餘弦相似度)
        query_embedding = get_model().encode([query])[0]
        semantic_scores = np.dot(question_embeddings, query_embedding)
        # 兩者加權平均（可調整權重）
        combined_scores = 0.7 * np.array(bm25_scores) + 0.3 * semantic_scores
        # 1. 篩選出超過基本閾值的結果
        above_threshold_indices = [
            i for i, score in enumerate(combined_scores) if score >= threshold
        ]
        
        if not above_threshold_indices:
            return []
        # 2. 按照綜合分數排序
        sorted_indices = sorted(
            above_threshold_indices, key=lambda i: combined_scores[i], reverse=True
        )
        
        high_score_indices = [
            i for i in sorted_indices if combined_scores[i] >= high_threshold
        ]
        
        result = []
        if len(high_score_indices) >= 2:
            # 如果有兩個或以上高分結果，返回前n個
            result = [
                {
                    "question": questions_in_sheet[i],
                    "answer": answers_in_sheet[i],
                    "bm25_score": float(bm25_scores[i]),
                    "semantic_score": float(semantic_scores[i]),
                    "combined_score": float(combined_scores[i]),
                }
                for i in high_score_indices[:n]
            ]
            threading.Thread(
                target=record_question_for_answer,
                args=(questions_in_sheet[high_score_indices[0]],),
            ).start()
        else:
            # 如果沒有或只有一個高分結果，只返回最高分的一個
            i = sorted_indices[0]
            result = [
                {
                    "question": questions_in_sheet[i],
                    "answer": answers_in_sheet[i],
                    "bm25_score": float(bm25_scores[i]),
                    "semantic_score": float(semantic_scores[i]),
                    "combined_score": float(combined_scores[i]),
                }
            ]
            threading.Thread(
                target=record_question_for_answer,
                args=(questions_in_sheet[sorted_indices[0]],),
            ).start()
        
        return result
    except Exception as e:
        print(f"Error in retrieve_top_n: {str(e)}")
        return []

###############################################################################
# LLM AND RESPONSE PROCESSING
###############################################################################

def reply_by_LLM(finalanswer, model):
    """使用LLM生成自然語言回覆"""
    try:
        prompt = f"""你是知識問答客服，請將{ finalanswer }直接轉成自然語言。
        ##條件
        1.口氣禮貌親切簡潔，像是和使用者對話
        2.若finalanswer為空[]，則回覆:此問題目前找不到合適解答，請聯絡積慧幫忙協助
        3.若finalanswer不為空[]，最後請換行後加一句:若此答案無法解決您問題，請換個問題再問一次或是聯絡積慧幫忙協助
        4.不要解釋以上回覆條件，直接回覆答案
        5.不要反問使用者
        """
        answer_in_human = model.generate_content(prompt)
        return answer_in_human
    except Exception as e:
        print(f"Error in reply_by_LLM: {str(e)}")
        return None

def extract_chinese_results_new(response):
    """從模型回應中提取中文內容"""
    try:
        text_content = response.candidates[0].content.parts[0].text
        
        if "\\u" in text_content:
            decoded_text = text_content.encode().decode("unicode_escape")
            return decoded_text
        
        return text_content
    except (AttributeError, IndexError, UnicodeError):
        return ""

def find_closest_question_and_llm_reply(query):
    """主要的問答處理函數"""
    try:
        top_matches = retrieve_top_n(query)
        if not top_matches:
            return {
                "answer": "目前找不到合適的答案，請再試一次或換個問法",
                "top_matches": [],
            }
        
        answers_only = [match["answer"] for match in top_matches]
        result = reply_by_LLM(answers_only, generation_model)
        answer_to_line = extract_chinese_results_new(result)
        return {"answer": answer_to_line, "top_matches": top_matches}
    
    except Exception as e:
        print(f"Error in find_closest_question_and_llm_reply: {str(e)}")
        return {
            "answer": "此問題目前找不到合適解答，請聯絡積慧幫忙協助",
            "top_matches": [],
        }

###############################################################################
# DATA RETRIEVAL FUNCTIONS
###############################################################################

def get_top_questions():
    """獲取熱門問題前5名"""
    try:
        ranking_ws = sheet.worksheet("title", "熱門排行")
        print("Found '熱門排行' worksheet.")
    except pygsheets.WorksheetNotFound:
        print("熱門排行 worksheet not found.")
        return []
    
    top_ranking_records = ranking_ws.get_all_records()[:5]
    top_questions = []
    
    main_ws = sheet.worksheet("title", "表單回應")
    main_records = main_ws.get_all_records()
    
    for record in top_ranking_records:
        full_question = next(
            (item for item in main_records if item["問題描述"] == record["項目"]), None
        )
        if full_question:
            top_questions.append({
                "排名": record["排名"],
                "項目": record["項目"],
                "問題描述": full_question["問題描述"],
                "解決方式": full_question["解決方式"],
            })
    
    print(f"Top 5 questions with descriptions: {top_questions}")
    return top_questions

def get_unique_categories():
    """獲取唯一問題分類"""
    try:
        main_ws = sheet.worksheet("title", "表單回應")
        categories_column = main_ws.get_col(2)
        unique_categories = sorted(
            list(set(cat.strip() for cat in categories_column[1:] if cat.strip()))
        )
        
        print(f"Found {len(unique_categories)} unique categories: {unique_categories}")
        return unique_categories
    except Exception as e:
        print(f"Error in get_unique_categories: {str(e)}")
        return []

def get_questions_by_category(category):
    """根據分類獲取問題"""
    try:
        main_ws = sheet.worksheet("title", "表單回應")
        all_data = main_ws.get_all_values()
        
        questions = []
        for row in all_data[1:]:
            if len(row) > 2 and row[1].strip() == category.strip():
                question_text = row[2].strip()
                if question_text:
                    questions.append({
                        "問題描述": question_text,
                        "解決方式": "",
                    })
        
        print(f"Total {len(questions)} questions found for category '{category}'")
        return questions
    except Exception as e:
        print(f"Error in get_questions_by_category: {str(e)}")
        return []

def find_solution_by_click_question(question_text):
    """找對應問題的解決方式"""
    try:
        main_ws = sheet.worksheet("title", "表單回應")
        all_data = main_ws.get_all_values()
        
        for row in all_data[1:]:
            if len(row) > 3 and row[2].strip() == question_text.strip():
                solution = row[3].strip()
                print(f"Found solution for question '{question_text}': {solution}")
                return solution
        
        print(f"No solution found for question '{question_text}'")
        return None
    except Exception as e:
        print(f"Error in find_solution_by_question: {str(e)}")
        return None

def get_oil_points_column_a():
    """獲取中油點數資料"""
    if not cpc_list or len(cpc_list) == 0:
        return "中油點數表單的 A 欄沒有資料。"
    
    return "\n".join(cpc_list)

###############################################################################
# LOGGING FUNCTIONS
###############################################################################

def record_question(user_id, user_input):
    """記錄用戶問題到統計紀錄"""
    gc = pygsheets.authorize(service_account_file='service_account_key.json')
    sheet = gc.open_by_url(os.environ.get("GOOGLESHEET_URL"))
    try:
        profile = line_bot_api.get_profile(user_id)
        user_name = profile.display_name
        print(f"Fetched user profile: {user_name}")
    except LineBotApiError as e:
        user_name = "Unknown"
        print(f"Error getting user profile: {e}")
    
    try:
        stats_ws = sheet.worksheet("title", "統計紀錄")
        print("Found '統計紀錄' worksheet.")
    except pygsheets.WorksheetNotFound:
        stats_ws = sheet.add_worksheet("統計紀錄")
        stats_ws.update_row(1, ["時間", "使用者ID", "使用者名稱", "詢問文字"])
        print("Created '統計紀錄' worksheet.")
    
    timestamp = datetime.now(GMT_8).strftime("%Y-%m-%d %H:%M:%S")
    record_data = [timestamp, user_id, user_name, user_input]
    stats_ws.insert_rows(row=1, values=record_data, inherit=True)
    print(f"Recorded question: {record_data}")

def record_question_for_answer(question_for_answer):
    """記錄回答問題到回答工作表"""
    gc = pygsheets.authorize(service_account_file='service_account_key.json')
    sheet = gc.open_by_url(os.environ.get("GOOGLESHEET_URL"))
    try:
        reply_ws = sheet.worksheet("title", "回答")
        print("Found '回答' worksheet.")
    except pygsheets.WorksheetNotFound:
        reply_ws = sheet.add_worksheet("回答")
        reply_ws.update_row(1, ["時間", "問題"])
        print("Created '回答' worksheet.")
    
    timestamp = datetime.now(GMT_8).strftime("%Y-%m-%d %H:%M:%S")
    record_data = [timestamp, question_for_answer]
    reply_ws.insert_rows(row=1, values=record_data, inherit=True)
    print(f"Recorded question: {record_data}")

###############################################################################
# UI AND FLEX MESSAGE FUNCTIONS
###############################################################################

def create_category_and_common_features():
    """生成分類選擇的Flex Message"""
    print("Generating category and common features message.")
    categories = get_unique_categories()
    category_bubble = BubbleContainer(
        body=BoxComponent(
            layout="vertical",
            contents=[
                TextComponent(
                    text="請選擇問題分類", weight="bold", size="xl", margin="md"
                )
            ]
            + [
                TextComponent(
                    text=f"{idx + 1}. {category}",
                    size="md",
                    color="#4682B4",
                    wrap=True,
                    margin="md",
                    action=MessageAction(label=category, text=f"問題分類: {category}"),
                )
                for idx, category in enumerate(categories[:10])
            ],
        )
    )
    
    return FlexSendMessage(
        alt_text="請選擇問題分類",
        contents=CarouselContainer(contents=[category_bubble]),
    )

def create_flex_message(title, items, item_type="category", start_index=1):
    """生成Flex Message以顯示搜尋結果或分類選項"""
    bubbles = []
    for i in range(0, len(items), 10):
        bubble_contents = [
            TextComponent(text=title, weight="bold", size="xl", margin="md")
        ]
        
        for idx, item in enumerate(items[i : i + 10], start=start_index):
            label_text = (
                f"{idx}. {item['問題描述'] if item_type == 'question' else item}"
            )
            action_text = (
                f"問題: {item['問題描述']}"
                if item_type == "question"
                else f"問題分類: {item}"
            )
            
            bubble_contents.append(
                TextComponent(
                    text=label_text,
                    size="md",
                    color="#4682B4",
                    wrap=True,
                    margin="md",
                    action=MessageAction(label=label_text[:20], text=action_text),
                )
            )
        
        bubble_contents.append(SeparatorComponent(margin="md"))
        bubble_contents.append(
            TextComponent(
                text="🔙 問題分類",
                weight="bold",
                color="#228B22",
                wrap=True,
                action=MessageAction(label="問題分類", text="返回問題分類"),
            )
        )
        
        bubbles.append(
            BubbleContainer(
                body=BoxComponent(layout="vertical", contents=bubble_contents)
            )
        )
        start_index += 10
    
    print(f"Generated Flex Message with title '{title}' and {len(bubbles)} bubbles.")
    return (
        FlexSendMessage(
            alt_text="請選擇分類或問題描述",
            contents=CarouselContainer(contents=bubbles),
        )
        if bubbles
        else TextSendMessage(text="找不到符合條件的資料。")
    )

def build_flex_response(answer, conversation_id):
    """建立包含回饋按鈕的Flex回覆"""
    return FlexSendMessage(
        alt_text="回覆與回饋",
        contents={
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {"type": "text", "text": answer, "wrap": True},
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "margin": "md",
                        "contents": [
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "👍",
                                    "data": f"feedback=thumbs_up&conv_id={conversation_id}",
                                },
                                "height": "sm",
                                "flex": 1,
                            },
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "👎",
                                    "data": f"feedback=thumbs_down&conv_id={conversation_id}",
                                },
                                "height": "sm",
                                "flex": 1,
                            },
                        ],
                    },
                ],
            },
        },
    )

###############################################################################
# LINE BOT EVENT HANDLERS
###############################################################################

@app.route("/callback", methods=["POST"])
def callback(request):
    print(f"Version Code: {VERSION_CODE}")
    
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    print("Request body:", body)
    
    try:
        payload = json.loads(body)
        if payload.get("destination") != ALLOWED_DESTINATION:
            print("Invalid destination.")
            return "Forbidden", 403
    except Exception as e:
        print("Payload parsing error:", e)
        return "Bad Request", 400
    
    try:
        handler.handle(body, signature)
        print("Message handled successfully.")
    except InvalidSignatureError as e:
        print("InvalidSignatureError:", e)
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text
    user_id = event.source.user_id
    
    if user_input.startswith("知識寶典") or user_input.startswith("返回問題分類"):
        reply = create_category_and_common_features()
        print("Displayed category and common features message.")
    
    elif user_input.startswith("問題分類:"):
        category = user_input.replace("問題分類:", "", 1).strip()
        print(f"Processing category request: '{category}'")
        
        questions = get_questions_by_category(category)
        
        if questions:
            print(f"Found {len(questions)} questions for category '{category}'")
            reply = create_flex_message(f"{category} - 問題列表", questions, "question")
        else:
            print(f"No questions found for category '{category}'")
            reply = TextSendMessage(
                text=f"找不到「{category}」分類的相關問題。請確認分類名稱是否正確。"
            )
    
    elif user_input.startswith("問題:"):
        question = user_input.replace("問題:", "", 1).strip()
        print(f"Looking for solution to question: '{question}'")
        
        solution = find_solution_by_click_question(question)
        
        if solution:
            reply_contents = [
                TextComponent(text="解決方式", weight="bold", size="lg", margin="md"),
                TextComponent(
                    text=solution, size="sm", color="#6A5ACD", wrap=True, margin="md"
                ),
                SeparatorComponent(margin="md"),
                TextComponent(
                    text="🔙 返回問題分類",
                    weight="bold",
                    color="#228B22",
                    wrap=True,
                    margin="md",
                    action=MessageAction(label="返回問題分類", text="返回問題分類"),
                ),
            ]
            
            reply = FlexSendMessage(
                alt_text="解決方式",
                contents=BubbleContainer(
                    body=BoxComponent(
                        layout="vertical", contents=reply_contents, padding_all="xl"
                    )
                ),
            )
            print(f"Displayed solution for question: {question}")
        else:
            reply = TextSendMessage(text="找不到該問題的解決方式。")
            print(f"No solution found for question: {question}")
    
    elif user_input == "熱門查詢":
        top_questions = get_top_questions()
        if top_questions:
            reply = create_flex_message("熱門查詢 - Top 5 問題", top_questions, "question")
        else:
            reply = TextSendMessage(text="目前沒有熱門排行記錄。")
        print("Displayed top 5 questions.")
    
    elif user_input == "查中油點數":
        oil_points_message = get_oil_points_column_a()
        reply = TextSendMessage(text=oil_points_message)
        print("Displayed '中油兌換點數' column A.")
    
    else:
        try:
            result_bundle = find_closest_question_and_llm_reply(user_input)
            conversation_id = f"conv_{user_id}_{int(time.time())}"
            reply = build_flex_response(result_bundle["answer"], conversation_id)
            print(f"Show LLM answer for question: {user_input}")
            
            if result_bundle["top_matches"]:
                top1 = result_bundle["top_matches"][0]
                
                db.collection("conversations").add({
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "question": user_input,
                    "answer": result_bundle["answer"],
                    "matched_question": top1["question"],
                    "bm25_score": top1["bm25_score"],
                    "semantic_score": top1["semantic_score"],
                    "combined_score": top1["combined_score"],
                    "model_version": VERSION_CODE,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
        
        except Exception as e:
            print(f"Error in find_closest_question_and_llm_reply: {str(e)}")
            reply = TextSendMessage(text="機器人暫時無法使用，請聯絡積慧幫忙協助")
    
    try:
        line_bot_api.reply_message(event.reply_token, reply)
        print("Reply sent successfully.")
    except LineBotApiError as e:
        print(f"Failed to send reply: {e}")
    
    # 非同步記錄用戶提問
    threading.Thread(target=record_question, args=(user_id, user_input)).start()

@handler.add(PostbackEvent)
def handle_postback(event):
    """處理用戶回饋"""
    data = event.postback.data
    params = dict(x.split("=") for x in data.split("&"))
    feedback_type = params.get("feedback")
    conversation_id = params.get("conv_id")
    user_id = event.source.user_id
    
    db.collection("feedback").add({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "feedback_type": feedback_type,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text="感謝您的回饋 🙏")
    )

###############################################################################
# MAIN APPLICATION
###############################################################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on port {port}")
    app.run(host="0.0.0.0", port=port)