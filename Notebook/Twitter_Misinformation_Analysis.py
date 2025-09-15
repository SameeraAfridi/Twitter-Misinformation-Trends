# =========================================================
# Twitter Misinformation Trends Analysis
# =========================================================

# --- 0. Install dependencies (for Colab) ---
!pip install pandas numpy matplotlib seaborn nltk wordcloud vaderSentiment plotly scikit-learn emoji

# --- 1. Imports ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import io
from google.colab import files

# download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# --- 2. Upload CSV ---
uploaded = files.upload()
kaggle_csv = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[kaggle_csv]), low_memory=False)
print("✅ Loaded Kaggle CSV. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- 3. Handle missing text ---
if not any("text" in c.lower() for c in df.columns):
    print("⚠️ No text column found. Creating synthetic tweets for demo...")
    df['text'] = df.apply(
        lambda row: f"This is a {row['account.type']} account classified as {row['class_type']} spreading info about elections and AI.",
        axis=1
    )
else:
    text_col = [c for c in df.columns if "text" in c.lower()][0]
    df = df.rename(columns={text_col: "text"})

# --- 4. Cleaning ---
def clean_tweet_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)             # remove @mentions
    text = re.sub(r'#', '', text)                # keep hashtag words
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  # remove emojis/punct
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df['text_clean'] = df['text'].apply(clean_tweet_text)

# --- 5. Word Cloud ---
all_text = " ".join(df['text_clean'].dropna().tolist())
wc = WordCloud(width=1200, height=600, background_color='white', stopwords=STOPWORDS).generate(all_text)
plt.figure(figsize=(14,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud: Top words in Tweets')
plt.show()

# --- 6. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    return analyzer.polarity_scores(text)

df['vader_scores'] = df['text_clean'].apply(vader_sentiment)
df['compound'] = df['vader_scores'].apply(lambda x: x['compound'])

def sentiment_label(c):
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"

df['sentiment'] = df['compound'].apply(sentiment_label)

# Sentiment distribution
sent_counts = df['sentiment'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
fig = px.bar(x=sent_counts.index, y=sent_counts.values,
             labels={'x':'Sentiment','y':'Count'},
             title='Tweet Sentiment Distribution')
fig.show()

# --- 7. "Misinformation Spread" (Fake vs Real accounts) ---
if "account.type" in df.columns:
    spread = df.groupby(["account.type","sentiment"]).size().reset_index(name="count")
    fig = px.bar(spread, x="account.type", y="count", color="sentiment",
                 title="Misinformation Spread by Account Type (Demo)",
                 barmode="stack")
    fig.show()

# --- 8. Save outputs ---
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/processed_tweets.csv", index=False)
wc.to_file("outputs/wordcloud.png")
print("✅ Saved processed data and wordcloud in outputs/")
