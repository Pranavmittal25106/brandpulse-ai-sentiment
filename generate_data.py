"""
generate_data.py
-----------------
Generates a realistic, messy, Twitter-style dataset for the BrandPulse AI
sentiment analysis project (Positive / Negative / Neutral).

NOTE ON DATA SOURCE:
This sandbox environment cannot reach Kaggle or the Sentiment140 hosting
servers, so this script builds a large, template-driven synthetic tweet
corpus that mimics real tweet noise: URLs, @mentions, #hashtags, emojis,
elongated words, typos, retweet markers, and mixed casing. If you have
access to the real Sentiment140 dataset or the Twitter US Airline
Sentiment dataset (Kaggle), simply replace `raw_tweets.csv` in this folder
with that file (columns: `text`, `sentiment`) and re-run the notebooks --
no other code changes are required.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

brands = ["@BrandPulseAir", "@SkyLineAir", "@JetNow", "@CloudCarrier", "@FlyNova"]
products = ["flight", "app", "customer service", "delivery", "in-flight wifi",
            "seat", "checkin process", "baggage handling", "support team", "website"]

positive_templates = [
    "Just had the best experience with {brand}! The {product} was amazing, totally worth it {emoji}",
    "Huge shoutout to {brand} for fixing my {product} issue so fast!! {emoji} #CustomerService",
    "Wow {brand} really outdid themselves this time, {product} was smooth af {emoji}",
    "I'm so impressed with {brand}, their {product} exceeded my expectations {emoji}",
    "Thank you {brand}!! You made my day, the {product} was perfect {emoji} #love",
    "honestly did not expect {brand} to be this good, {product} 10/10 {emoji}",
    "Can we talk about how great {brand}'s {product} is?? Blown away {emoji}",
    "{brand} customer support just solved my problem in 2 mins, incredible {emoji}",
    "loving the new {product} from {brand}, definitely recommending to friends {emoji}",
    "props to {brand} for such a smooth {product} experience today {emoji} great job",
]

negative_templates = [
    "worst {product} ever with {brand}, absolutely furious right now {emoji}",
    "{brand} you've got to be kidding me, {product} was a disaster {emoji} #fail",
    "I waited 4 hours just to get my {product} sorted with {brand}, ridiculous",
    "never using {brand} again, {product} completely ruined my day {emoji}",
    "so done with {brand}'s terrible {product}, this is unacceptable {emoji}",
    "{brand} customer service hung up on me twice, {product} still broken {emoji}",
    "extremely disappointed in {brand}, the {product} was a complete mess {emoji}",
    "cant believe {brand} messed up my {product} again smh {emoji}",
    "{brand} is a joke, {product} delayed with zero communication {emoji} #angry",
    "worst decision trusting {brand} with my {product}, total nightmare {emoji}",
]

neutral_templates = [
    "just booked a {product} with {brand}, will update after the trip",
    "{brand} sent me an email about {product} changes, reading through it now",
    "does anyone know if {brand}'s {product} is working today? checking status",
    "{brand} {product} policy update just dropped, here are the details",
    "flying with {brand} next week, first time trying their {product}",
    "{brand} released a statement regarding {product} maintenance schedule",
    "curious about {brand}'s {product}, anyone have experience with it?",
    "{brand} {product} page shows standard pricing for this month",
    "reminder: {brand} {product} check-in opens 24 hours before departure",
    "{brand} just tweeted an update about the {product}, nothing major",
    "{brand}'s {product} was okay I guess, nothing special either way",
    "not sure how I feel about the new {product} from {brand}, still deciding",
    "{brand} {product} seems fine so far, will see how it goes",
    "mixed feelings about {brand}'s {product}, some parts good some not",
]

# Sarcastic negative templates: use positive-sounding words to express negative
# sentiment. These are intentionally the hardest examples in the dataset --
# a bag-of-words model (TF-IDF) tends to see "great"/"amazing" and lean positive,
# while a sequence model has a better (but still imperfect) shot at the context.
sarcastic_negative_templates = [
    "oh great, another {product} delay from {brand}, exactly what I needed today",
    "wow {brand}, amazing job breaking my {product} again, so impressed (not)",
    "love how {brand} 'fixed' my {product}, still broken lol thanks so much",
    "great job {brand}, {product} is fantastic if you enjoy waiting forever",
    "yeah {brand}'s {product} is just perfect, said no one ever",
    "thanks a lot {brand} for the 'wonderful' {product} experience today",
    "sure {brand}, tell me again how great your {product} is, real convincing",
    "fantastic, {brand} lost my {product} request, love starting my day like this",
]

# Generic, non-branded review-style sentences -- these broaden vocabulary beyond
# the branded-tweet phrasing so the model generalizes to plain customer feedback
# like "The service was amazing!" rather than only recognizing brand-hashtag patterns.
generic_positive = [
    "The {product} was amazing!",
    "Absolutely loved the {product}, would recommend to anyone.",
    "This {product} exceeded all my expectations, wonderful experience.",
    "What a fantastic {product}, I'm genuinely impressed.",
    "The {product} was excellent from start to finish.",
    "Really happy with the {product}, everything went smoothly.",
    "Superb {product}! Couldn't have asked for better.",
    "The {product} was great, exactly what I hoped for.",
    "I'm thrilled with how the {product} turned out.",
    "The {product} was outstanding, five stars all the way.",
]
generic_negative = [
    "I waited 4 hours just to get a cold {product}.",
    "The {product} was terrible, I want a refund.",
    "Never again, the {product} was a complete disappointment.",
    "The {product} was awful and the staff didn't care at all.",
    "Waited forever for the {product}, absolutely unacceptable.",
    "The {product} broke almost immediately, total waste of money.",
    "Such a bad experience with the {product}, would not recommend.",
    "The {product} was way below what I expected, very disappointing.",
    "Horrible {product}, nothing worked as it should.",
    "The {product} was a disaster from beginning to end.",
]

pos_emojis = ["😊", "🎉", "❤️", "🙌", "😍", ":)", "!!", ""]
neg_emojis = ["😡", "😠", "💔", "🙄", ":(", "...", "!!", ""]
neu_emojis = ["", "", "", ""]

def add_noise(text):
    """Add realistic tweet noise: urls, mentions, hashtags, RT markers, typos."""
    if random.random() < 0.35:
        text = "RT " + text
    if random.random() < 0.4:
        text += " http://t.co/" + "".join(random.choices("abcdefgh12345", k=8))
    if random.random() < 0.3:
        text += " #" + random.choice(["travel", "customerexperience", "brand",
                                       "review", "service", "today"])
    if random.random() < 0.25:
        text += " @user" + str(random.randint(100, 999))
    if random.random() < 0.15:
        # elongate a word for emphasis (common tweet noise)
        words = text.split()
        if words:
            idx = random.randrange(len(words))
            w = words[idx]
            if len(w) > 3 and w.isalpha():
                ch = w[-1]
                words[idx] = w + ch * random.randint(1, 3)
            text = " ".join(words)
    if random.random() < 0.2:
        text = text.upper() if random.random() < 0.3 else text
    return text

def make_generic_rows(templates, label, n):
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        product = random.choice(products)
        text = t.format(product=product)
        # lighter noise -- generic reviews are cleaner than tweets, but still add some
        if random.random() < 0.15:
            text += " #" + random.choice(["review", "feedback", "experience"])
        if random.random() < 0.1:
            text = text.upper()
        rows.append(text)
    return pd.DataFrame({"text": rows, "sentiment": [label] * n})

def make_rows(templates, emojis, label, n):
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        brand = random.choice(brands)
        product = random.choice(products)
        emoji = random.choice(emojis)
        text = t.format(brand=brand, product=product, emoji=emoji)
        text = add_noise(text)
        rows.append(text)
    return pd.DataFrame({"text": rows, "sentiment": [label] * n})

n_per_class = 1600
n_sarcastic = 500  # hard examples, all labeled "negative"

df_pos = make_rows(positive_templates, pos_emojis, "positive", n_per_class)
df_neg = make_rows(negative_templates, neg_emojis, "negative", n_per_class)
df_neu = make_rows(neutral_templates, neu_emojis, "neutral", n_per_class)
df_sarcastic = make_rows(sarcastic_negative_templates, pos_emojis + neg_emojis, "negative", n_sarcastic)
df_generic_pos = make_generic_rows(generic_positive, "positive", 500)
df_generic_neg = make_generic_rows(generic_negative, "negative", 500)

df = pd.concat([df_pos, df_neg, df_neu, df_sarcastic, df_generic_pos, df_generic_neg], ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Inject a small amount of label noise (~4%) to simulate real-world annotation
# inconsistency -- real Twitter sentiment datasets are never perfectly clean,
# and this keeps both models from trivially hitting 100% accuracy.
NOISE_RATE = 0.04
noise_idx = df.sample(frac=NOISE_RATE, random_state=7).index
all_labels = ["positive", "negative", "neutral"]
for idx in noise_idx:
    current = df.loc[idx, "sentiment"]
    choices = [l for l in all_labels if l != current]
    df.loc[idx, "sentiment"] = random.choice(choices)

# Fake realistic timestamps over the last 24 hours for the dashboard's "trend over time" feature
now = datetime(2026, 7, 13, 12, 0, 0)
timestamps = [now - timedelta(minutes=random.randint(0, 24 * 60)) for _ in range(len(df))]
df["created_at"] = sorted(timestamps)

df.to_csv("/home/claude/brandpulse_ai/data/twitter_sentiment_data.csv", index=False)
print(f"Saved {len(df)} rows to twitter_sentiment_data.csv")
print(df["sentiment"].value_counts())
print(df.head(5))
