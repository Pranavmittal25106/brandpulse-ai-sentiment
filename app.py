"""
app.py — BrandPulse AI Sentiment Dashboard
--------------------------------------------
Streamlit app that loads the saved classical (TF-IDF + Logistic Regression) and
deep learning (LSTM) models and visualizes predictions on new/incoming tweets.

Run with:  streamlit run app.py
"""

import re
import json
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------------
# Page config & caching
# ----------------------------------------------------------------------------
st.set_page_config(page_title="BrandPulse AI — Sentiment Dashboard", layout="wide", page_icon="📊")

for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(pkg if "/" in pkg else f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
URL_RE = re.compile(r"http\S+|www\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
RT_RE = re.compile(r"^rt\b", flags=re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = RT_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_lemmatize(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


@st.cache_resource
def load_assets():
    classical_model = joblib.load("models/classical_model.joblib")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    lstm_model = load_model("models/lstm_model.keras")
    label_encoder = joblib.load("models/label_encoder.joblib")
    with open("models/tokenizer.json") as f:
        tokenizer = tokenizer_from_json(f.read())
    with open("models/classical_metrics.json") as f:
        classical_metrics = json.load(f)
    with open("models/lstm_metrics.json") as f:
        lstm_metrics = json.load(f)
    return classical_model, tfidf, lstm_model, label_encoder, tokenizer, classical_metrics, lstm_metrics


@st.cache_data
def load_data():
    df = pd.read_csv("data/twitter_sentiment_data.csv", parse_dates=["created_at"])
    return df


(classical_model, tfidf, lstm_model, label_encoder,
 keras_tokenizer, classical_metrics, lstm_metrics) = load_assets()
df = load_data()
MAX_LEN = lstm_metrics["max_len"]

SENTIMENT_COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

def predict_classical(text: str) -> str:
    clean = tokenize_lemmatize(clean_tweet(text))
    vec = tfidf.transform([clean])
    return classical_model.predict(vec)[0]

def predict_lstm(text: str):
    clean = tokenize_lemmatize(clean_tweet(text))
    seq = keras_tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = lstm_model.predict(pad, verbose=0)[0]
    label = label_encoder.inverse_transform([np.argmax(probs)])[0]
    return label, dict(zip(label_encoder.classes_, probs.round(3)))

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("📊 BrandPulse AI")
st.sidebar.caption("Sentiment Analysis on Twitter Data — NLP Internship Project")
model_choice = st.sidebar.radio(
    "Model for live inference",
    ["Classical (TF-IDF + " + classical_metrics["best_model_name"] + ")", "Deep Learning (LSTM)"],
)
st.sidebar.markdown("---")
st.sidebar.metric("Classical Model F1", f"{classical_metrics['all_results'][classical_metrics['best_model_name']]['f1']:.3f}")
st.sidebar.metric("LSTM Model F1", f"{lstm_metrics['f1']:.3f}")
st.sidebar.markdown("---")
st.sidebar.caption("Case Study: automated brand-sentiment monitoring so marketing "
                    "teams can filter the noise on Twitter/X and understand the "
                    "Voice of the Customer instantly.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("BrandPulse AI — Live Sentiment Dashboard")
st.caption("Classify tweets as Positive / Negative / Neutral in real time and track brand sentiment trends.")

tab_live, tab_trend, tab_infer, tab_compare = st.tabs(
    ["🔴 Live Stream", "📈 Trend Over Time", "🔎 Test Custom Input", "⚖️ Model Comparison"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Live Stream simulation
# ----------------------------------------------------------------------------
with tab_live:
    st.subheader("Simulated Live Tweet Feed")
    col_a, col_b = st.columns([2, 1])

    if "stream_idx" not in st.session_state:
        st.session_state.stream_idx = 0
        st.session_state.stream_sample = df.sample(n=min(300, len(df)), random_state=1).reset_index(drop=True)

    with col_b:
        n_new = st.slider("Tweets per refresh", 1, 10, 3)
        auto = st.toggle("Auto-play", value=False)
        if st.button("▶️ Push next batch") or auto:
            st.session_state.stream_idx = min(
                st.session_state.stream_idx + n_new, len(st.session_state.stream_sample)
            )

    visible = st.session_state.stream_sample.iloc[: max(st.session_state.stream_idx, n_new)]

    with col_a:
        st.caption(f"Showing {len(visible)} incoming tweets")
        for _, row in visible.tail(8).iloc[::-1].iterrows():
            label = row["sentiment"]
            color = SENTIMENT_COLORS.get(label, "#999")
            st.markdown(
                f"<div style='padding:8px 12px;border-left:4px solid {color};"
                f"background:rgba(255,255,255,0.03);margin-bottom:6px;border-radius:4px;'>"
                f"<span style='color:{color};font-weight:600;text-transform:uppercase;font-size:11px;'>{label}</span>"
                f"<br>{row['text']}</div>",
                unsafe_allow_html=True,
            )

    # Live sentiment distribution pie chart (Feature: Sentiment Distribution)
    st.subheader("Sentiment Distribution (% of Positive vs. Negative vs. Neutral)")
    if len(visible) > 0:
        counts = visible["sentiment"].value_counts().reindex(["positive", "negative", "neutral"]).fillna(0)
        fig_pie = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index, color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    if auto:
        time.sleep(1.2)
        st.rerun()

# ----------------------------------------------------------------------------
# TAB 2 — Trend over time
# ----------------------------------------------------------------------------
with tab_trend:
    st.subheader("Sentiment Trend — Last 24 Hours")
    trend_df = df.copy()
    trend_df["hour"] = trend_df["created_at"].dt.floor("H")
    hourly = trend_df.groupby(["hour", "sentiment"]).size().reset_index(name="count")
    fig_trend = px.line(
        hourly, x="hour", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS, markers=True,
        labels={"hour": "Time", "count": "Number of Tweets"},
    )
    fig_trend.update_layout(legend_title_text="Sentiment")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Time-series view of how sentiment volume changes over the last 24 hours "
               "— useful for spotting spikes in negative sentiment (e.g. an outage or PR issue).")

# ----------------------------------------------------------------------------
# TAB 3 — Custom inference
# ----------------------------------------------------------------------------
with tab_infer:
    st.subheader("Test With Your Own Text")
    default_examples = [
        "The service was amazing!",
        "I waited 4 hours just to get a cold burger.",
        "Great, another delay, exactly what I needed today.",
        "Just booked a flight, first time trying this airline.",
    ]
    user_text = st.text_area("Enter a tweet-like sentence:", value=default_examples[0], height=90)
    example_pick = st.selectbox("...or pick an example", [""] + default_examples)
    if example_pick:
        user_text = example_pick

    if st.button("Analyze Sentiment", type="primary"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classical (TF-IDF + " + classical_metrics["best_model_name"] + ")**")
            pred_c = predict_classical(user_text)
            st.markdown(f"### :{('green' if pred_c=='positive' else 'red' if pred_c=='negative' else 'gray')}[{pred_c.upper()}]")
        with col2:
            st.markdown("**Deep Learning (LSTM)**")
            pred_l, probs_l = predict_lstm(user_text)
            st.markdown(f"### :{('green' if pred_l=='positive' else 'red' if pred_l=='negative' else 'gray')}[{pred_l.upper()}]")
            st.bar_chart(pd.Series(probs_l))

# ----------------------------------------------------------------------------
# TAB 4 — Model comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Classical NLP vs. Deep Learning — Performance Report")
    best_name = classical_metrics["best_model_name"]
    rows = []
    for name, m in classical_metrics["all_results"].items():
        rows.append({"Model": f"{name} (Classical)", "Accuracy": m["accuracy"],
                      "Precision": m["precision"], "Recall": m["recall"], "F1-Score": m["f1"]})
    rows.append({"Model": "LSTM (Deep Learning)", "Accuracy": lstm_metrics["accuracy"],
                 "Precision": lstm_metrics["precision"], "Recall": lstm_metrics["recall"],
                 "F1-Score": lstm_metrics["f1"]})
    comp_df = pd.DataFrame(rows).set_index("Model").round(4)
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#1e5631"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("reports/confusion_matrix_classical.png", caption=f"Confusion Matrix — {best_name} (Classical)")
    with col2:
        st.image("reports/confusion_matrix_lstm.png", caption="Confusion Matrix — LSTM (Deep Learning)")

    st.markdown(
        "**Takeaway:** Both approaches perform similarly on this dataset. The classical "
        "TF-IDF model is fast, cheap to train, and easy to interpret (you can inspect "
        "feature weights directly). The LSTM captures word order and context, which "
        "matters most on sarcastic or ambiguous tweets, but costs more to train and is "
        "harder to explain. In production, a common pattern is to start with the "
        "classical baseline and only invest in deep learning once its interpretability "
        "cost is justified by a measurable accuracy gain on the hardest examples."
    )
