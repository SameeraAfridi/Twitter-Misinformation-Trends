# 🐦 TweepFake Analysis – Sentiment & Bot Detection

## 📌 Project Overview
This project analyzes tweets from the **TweepFake dataset** to study:
- **Sentiment (Positive / Negative / Neutral)**
- **Bot vs Human Account Distribution**
- **Most Common Words in Tweets (Word Cloud)**

The dataset contains tweets labeled as **human** or **bot**, making it useful for social media research and fake account detection.

---

## ⚙️ Features
1. **Data Cleaning**
   - Removed links, special characters, and stopwords.
   - Converted all text to lowercase.

2. **Word Cloud**
   - Visualized the most frequent words used in tweets.

3. **Sentiment Analysis**
   - Used **VADER Sentiment Analyzer** to classify tweets as:
     - Positive
     - Negative
     - Neutral

4. **Bot vs Human Distribution**
   - Compared the number of bot accounts vs human accounts.

---

## 📂 Dataset
- The dataset (`.csv`) includes:
  - `status_id` → Tweet text
  - `account.type` → Human or Bot
  - `class_type` → Extra classification labels

---

## 📊 Results
- **Word Cloud** → Shows the most common words in tweets.  
- **Sentiment Distribution** → Most tweets are **neutral**, fewer are positive/negative.  
- **Bot vs Human Accounts** → Dataset contains both human and bot tweets, visualized in a bar chart.  

👉<img width="709" height="381" alt="image" src="https://github.com/user-attachments/assets/3eda9587-e4bc-4d0e-93dd-747148b1c9b7" />
<img width="419" height="294" alt="image" src="https://github.com/user-attachments/assets/147cd676-f6e7-4c4a-b11c-f4f89fabfce2" />
<img width="416" height="316" alt="image" src="https://github.com/user-attachments/assets/2c777cf3-5bf4-4c57-a494-fb60632c6e7d" />


---

## 🚀 How to Run
1. Open the notebook in **Google Colab**.
2. Install dependencies:
   ```bash
   !pip install vaderSentiment
