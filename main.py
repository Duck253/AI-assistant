import requests
import os
import re
import json
import platform
import subprocess
import urllib.parse
import webbrowser
import joblib
import torch
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
from modules.scheduler import Scheduler
from modules.weather import get_weather
from modules.sentiment import analyze_sentiment
from modules.qa import answer_general_question
from modules.radio import stream_radio_vlc
from modules.finance import get_stock_price
from modules.calories import analyze_food_intake
from modules.music import play_song, recommend_song
from modules.medical import search_medical_condition
from utils.memory_manager import save_memory, load_memory, clear_memory
from modules.news import News

# Load NLP and intent classification model
tokenizer = AutoTokenizer.from_pretrained("./models/intent_model")
model = AutoModelForSequenceClassification.from_pretrained("./models/intent_model")

label_encoder = joblib.load("./models/intent_model/label_encoder.pkl")

nlp = spacy.load("en_core_web_sm")

# Intent fallback keywords
INTENTS = {
    "weather": ["weather", "forecast", "temperature"],
    "time": ["time", "clock"],
    "news": ["news", "headline", "update"],
    "play_song": ["play", "music"],
    "symptoms": ["symptom", "feeling", "ill", "sick", "condition"],
    "drug": ["drug", "medicine", "pill", "medication"],
    "exit": ["exit", "quit", "close"],
    "finance": ["finance", "stock", "price", "market"],
    "schedule": ["schedule", "remind", "meeting", "task"],
    "delete": ["delete", "remove", "cancel"],
    "question": ["who", "what", "when", "where", "how", "why"],
    "emotion_song": ["bored", "sad", "happy", "energetic", "chill"],
    "calories": ["ate", "calorie", "food"],
    "sentiment": ["feel", "emotion", "opinion"],
    "radio": ["radio", "vn"]
}


def predict_intent(text, threshold=0.1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    if confidence < threshold:
        return "unknown", confidence

    intent = label_encoder.inverse_transform([predicted_class])[0]
    return intent, confidence


def detect_intent(message):
    lowered = message.lower()
    for intent, keywords in INTENTS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    doc = nlp(lowered)
    for token in doc:
        for intent, keywords in INTENTS.items():
            if token.lemma_ in keywords:
                return intent
    return "unknown"


def ai_assistant():
    print("\n🤖 Smart AI Assistant Ready! Type 'exit' to quit.")
    schedule = Scheduler()
    print(schedule.daily_summary())

    conversation_history = []
    clear_memory()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "history":
            for i, pair in enumerate(conversation_history):
                print(f"\n🧠 {i+1}. You: {pair['user']}\n🤖 {pair['assistant']}")
            continue

        intent, confidence = predict_intent(user_input)
        print(f"[DEBUG] Predicted Intent: {intent} (Confidence: {confidence:.2f})")

        if intent == "unknown":
            print("🤔 I’m not quite sure what you mean. Can you rephrase?")
            continue

        response = ""
        if intent == "exit":
            print("👋 Goodbye!")
            break

        elif intent == "weather":
            city = input("Which city? ")
            response = get_weather(city)

        elif intent == "time":
            response = datetime.now().strftime("Current time: %H:%M:%S on %A, %d %B %Y")

        elif intent == "news":
            response = News.get_top_headlines()


        elif intent == "schedule":
            action = input("Do you want to [add], [view], or [delete] a task? ").strip().lower()
            if action == "add":
                response = schedule.add_schedule()
            elif action == "view":
                response = schedule.view_schedule()
            elif action == "delete":
                response = schedule.delete_schedule()
            else:
                response = "❌ Unknown action."

        elif intent == "symptoms":
            term = input("🩺 What symptom or condition? ")
            response = search_medical_condition(term)

        elif intent == "finance":
            symbol = input("📈 Enter stock symbol: ")
            response = get_stock_price(symbol)

        elif intent == "emotion_song":
            mood = input("🎧 Mood (happy/sad/energetic/chill): ").strip().lower()
            track, artist = recommend_song(mood)
            if track:
                play_song(track)
                response = f"🎵 '{track}' by {artist}"
            else:
                response = "❌ No match."

        elif intent == "play_song":
            query = input("🎶 What song? ")
            response = play_song(query)

        elif intent == "calories":
            food = input("🍕 What did you eat? ")
            response = analyze_food_intake(food)

        elif intent == "radio":
            stream_radio_vlc("http://media.kythuatvov.vn:7001/")
            response = "📻 VOV1 streaming..."

        elif intent == "sentiment":
            text = input("📝 Text to analyze: ")
            response = analyze_sentiment(text)

        elif intent == "question":
            response = answer_general_question(user_input)

        conversation_history.append({"user": user_input, "assistant": response})
        save_memory(conversation_history)
        print(response)


if __name__ == "__main__":
    ai_assistant()
