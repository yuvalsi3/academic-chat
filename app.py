import pandas as pd
import requests
import re
import os
from flask import Flask, request, jsonify
from flask import render_template


app = Flask(__name__)

GROQ_API_KEY = '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'
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'

df = pd.read_csv("universities.csv")
session_state = {"current_university": None}


def extract_university_llm(question, universities):
    system_prompt = """
You are an expert entity extractor.

You will receive a user's question and a list of institution names. 
Your job is to identify which institution (if any) the user is referring to, even if the name is misspelled, partially written, or contains typos.

Only return a single institution name exactly as written in the list. 
If you cannot confidently identify any, return 'None'.
"""
    university_list = "\n".join(universities)
    user_prompt = f"User question: {question}\nAvailable universities:\n{university_list}\n\nWhich institution is mentioned?"

    extracted = call_groq(system_prompt, user_prompt)
    extracted = extracted.strip()
    if "none" in extracted.lower():
        return None
    return extracted


def is_general_question(user_input):
    user_input = user_input.lower()
    general_keywords = ["which university", "best university", "compare", "comparison", "rank", "ranking", "universities", "better university"]
    return any(keyword in user_input for keyword in general_keywords)


def extract_keywords(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    keywords = [w for w in words if len(w) > 2]
    return keywords

def query_data(df, question, university=None):
    keywords = extract_keywords(question)
    def row_match(row):
        row_text = ' '.join(row.astype(str).str.lower())
        return any(kw in row_text for kw in keywords)
    filtered_df = df[df.apply(row_match, axis=1)]
    if university:
        filtered_df = filtered_df[filtered_df['Institution Name'] == university]
    return filtered_df.to_dict(orient="records")

def call_groq(system_prompt, user_prompt, chat_history=[]):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']

def optimize_question(user_question, chat_history):
    system_prompt = """
You are a university advisor assistant. Rewrite the user's question into a single short, precise, and optimized search query. Do not return multiple options or explanations.
"""
    return call_groq(system_prompt, f"Original question: '{user_question}'", chat_history).strip()

def generate_answer(user_question, data, chat_history):
    if not data:
        context = "No matching university data found."
    else:
        context = ""
        for record in data:
            context += "\nOption:\n"
            for key, value in record.items():
                if pd.isna(value) or str(value).strip() in ["", "nan", "None", "no need", "No requirement"]:
                    continue  # מדלגים על שדות ריקים
                context += f"- {key}: {value}\n"

    system_prompt = """
    You are a helpful and precise university advisor assistant.

    You receive structured university data, including institution name, degree name, city, admission requirements (such as Bagrut average, math requirement, English requirement, academic English level, psychometric score, composite score, degree schedule, and additional requirements).

    Use only the provided university data for your answers. Do not invent or assume any information that is not explicitly present in the dataset.

    - If the user asks for available degrees at a specific institution, respond only with the list of degree names available at that institution. Do not include any requirements or admission details.
    - If the user asks for the admission requirements of a specific degree (optionally within a university), return all available admission options for that degree and university. Number each option and display each field on a new line. Present every available admission path, listing all requirements exactly as provided. Never omit any requirement field.
    - When displaying admission requirements, do not include the degree duration (schedule) or tuition fees, unless the user explicitly requests them.
    - Only if the user explicitly requests to compare institutions or programs, provide for each institution: tuition fees (show the lowest tuition fee if multiple options exist), degree duration (schedule), and all admission requirements exactly as provided. After presenting the comparison, analyze the data and recommend which institution is most worthwhile based on: the lowest admission requirements, the lowest tuition fee, and the shortest degree duration.
    - Do not include any comparison or recommendation unless the user explicitly requests a comparison in the current question. Ignore any previous context or comparisons unless directly requested again in the current query.
    - Present only fields that have actual values in the data. If a field is empty — fully omit that field. Do not display the field name at all.
    - If the user asks about study duration (degree schedule), extract and display the exact value from the 'Degree Schedule' field.
    - Do not use bold, stars (**) or markdown. Just plain clean text.
    - If the user asks for general recommendations or rankings not present in the data, politely explain that you can only provide factual data from the database, not personal advice or rankings.
    - Always keep your answers short, clear, polite, and well-formatted.
    - You may assume the user is referring to the dataset provided. If no match is found for a user's request, politely state that no matching program was found.
    """

    full_prompt = f"User asked: {user_question}\n\nHere is institution data:\n{context}"
    return call_groq(system_prompt, full_prompt, chat_history).strip()

# API endpoint:
@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json
    user_input = data.get("message")
    chat_history = data.get("history", [])

    universities = df['Institution Name'].unique()
    university_mentioned = extract_university_llm(user_input, universities)

    if is_general_question(user_input):
        session_state["current_university"] = None
    else:
        session_state["current_university"] = university_mentioned

    optimized = optimize_question(user_input, chat_history)
    results = query_data(df, optimized, session_state.get("current_university"))
    answer = generate_answer(user_input, results, chat_history)

    if any(phrase in answer.lower() for phrase in
             ["i couldn't find", "i apologize", "no matching", "i don't", "cannot find", "not enough information", "no data", "sorry", "unable to answer", "unfortunately"]):
        session_state["current_university"] = None
        results = query_data(df, optimized, session_state.get("current_university"))
        answer = generate_answer(user_input, results, chat_history)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})

    return jsonify({"answer": answer, "history": chat_history})

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)