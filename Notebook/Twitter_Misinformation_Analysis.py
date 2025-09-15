# --- 1. Install missing dependencies ---
!pip install vaderSentiment

# --- 2. Imports ---
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 3. Upload CSV ---
uploaded = files.upload()
filename = list(uploaded.keys())[0]   # grab the uploaded filename

# --- 4. Load CSV ---
df = pd.read_csv(filename)   # use the uploaded file name directly

print("✅ Loaded TweepFake CSV. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Rename 'status_id' -> 'text' for consistency
if "status_id" in df.columns:
    df = df.rename(columns={"status_id": "text"})

# --- 5. Cleaning function ---
nltk.download('stopwords')

def clean_tweet_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove links
    text = re.sub(r"[^a-z\s]", "", text)        # keep only letters
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    return text

df["text_clean"] = df["text"].astype(str).apply(clean_tweet_text)

# --- 6. Word Cloud ---
all_words = " ".join(df["text_clean"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Tweets", fontsize=16)
plt.show()

# --- 7. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["text_clean"].apply(get_sentiment)

# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment", data=df, palette="Set2")
plt.title("Sentiment Distribution")
plt.show()

# --- 8. Bot vs Human Distribution ---
if "account.type" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="account.type", data=df, palette="coolwarm")
    plt.title("Bot vs Human Accounts")
    plt.show()

# --- 9. Save processed file (optional) ---
df.to_csv("tweets_processed.csv", index=False)
print("✅ Analysis complete. Processed file saved as tweets_processed.csv")
