import pandas as pd
import requests
import re
import os
from flask import Flask, request, jsonify, send_from_directory
from flask import render_template


app = Flask(__name__)

GROQ_API_KEY = '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'
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'

df = pd.read_csv("universities.csv")

def extract_keywords(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    keywords = [w for w in words if len(w) > 3]
    return keywords

def query_data(df, question):
    keywords = extract_keywords(question)
    def row_match(row):
        row_text = ' '.join(row.astype(str).str.lower())
        return any(kw in row_text for kw in keywords)
    filtered_df = df[df.apply(row_match, axis=1)]
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
            context += "\nUniversity:\n"
            for key, value in record.items():
                context += f"- {key}: {value}\n"

    system_prompt = """
    You are a helpful and precise university advisor assistant.

    You receive structured university data, including institution name, degree name, city, admission requirements (such as Bagrut average, math requirement, English requirement, academic English level, psychometric score, composite score, degree schedule, and additional requirements).

    Use only the provided university data for your answers. Do not invent or assume any information that is not explicitly present in the dataset.

    - If the user asks for available degrees at a specific institution, respond only with the list of degree names available at that institution. Do not include any requirements or admission details.
    - If the user asks for the admission requirements of a specific degree (optionally within a university), return all available admission options for that degree and university. Present every available admission path, listing all requirements exactly as provided. Never omit any requirement field.
    - If any data field is empty or missing for a specific option, you may display it as "No requirement" or "Not specified."
    - If the user asks about study duration (degree schedule), extract and display the exact value from the 'Degree Schedule' field.
    - Always keep your answers short, clear, polite, and well-formatted.
    - You may assume the user is referring to the dataset provided. If no match is found for a user's request, politely state that no matching program was found.
    """

    full_prompt = f"User asked: {user_question}\n\nHere is university data:\n{context}"
    return call_groq(system_prompt, full_prompt, chat_history).strip()

# API endpoint:
@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json
    user_input = data.get("message")
    chat_history = data.get("history", [])

    optimized = optimize_question(user_input, chat_history)
    results = query_data(df, optimized)
    answer = generate_answer(user_input, results, chat_history)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})

    return jsonify({"answer": answer, "history": chat_history})

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
