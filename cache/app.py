import streamlit as st
import pandas as pd
import nltk
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("omw-1.4")

st.set_page_config(page_title="Drug Review Sentiment Analyzer", layout="wide")

@st.cache_data
def load_data():
    file_path = "drugsComTrain_raw.csv"
    if not os.path.exists(file_path):
        st.error("Dataset not found. Please upload 'drugsComTrain_raw.csv'.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, usecols=["drugName", "condition", "review", "rating"])
        df.dropna(inplace=True)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["review"].str.strip() != ""]  # Remove empty reviews
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.warning("⚠ Dataset is empty. Please upload 'drugsComTrain_raw.csv'.")
    st.stop()

def categorize_rating(rating):
    return "Positive" if rating >= 7 else "Negative" if rating <= 4 else "Neutral"

def estimate_rating(reviews):
    if not reviews:
        return 5.0  # Default neutral rating
    estimated = np.mean([(TextBlob(review).sentiment.polarity + 1) * 4.5 + 1 for review in reviews])
    return np.clip(estimated, 1, 10)

def classify_sentiment(estimated_rating):
    return categorize_rating(estimated_rating)

df_filtered = df.dropna(subset=["condition", "review"])

st.sidebar.header("🔍 Filter Options")
selected_drug = st.sidebar.selectbox("Select a Drug", df["drugName"].unique())

constraints = [
    "Predict a patient’s condition based on reviews",
    "Estimate drug ratings from reviews",
    "Identify key elements that make reviews helpful",
    "Classify reviews as Positive, Neutral, or Negative",
]
selected_constraint = st.sidebar.selectbox("Select a Constraint", constraints)

st.sidebar.markdown(f"📌 Selected Constraint: **{selected_constraint}**")

df_filtered = df[df["drugName"] == selected_drug]

st.subheader(f"📊 Aggregated Analysis for **{selected_drug}**")

if df_filtered.empty:
    st.warning("⚠ No reviews found for the selected drug.")
    st.stop()

try:
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")  # Removed common stopwords
    X_tfidf = tfidf_vectorizer.fit_transform(df["review"])
    X_tfidf = csr_matrix(X_tfidf)  # Convert to sparse format
except ValueError:
    st.error("TF-IDF error: Empty vocabulary. Ensure dataset contains meaningful text.")
    st.stop()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["condition"].astype(str))

rus = RandomUnderSampler(random_state=42)
try:
    X_resampled, y_resampled = rus.fit_resample(X_tfidf, y_encoded)
except MemoryError:
    st.error("Memory limit exceeded. Try reducing dataset size or using a smaller feature set.")
    st.stop()

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced estimators to save memory
rf_classifier.fit(X_resampled, y_resampled)

def predict_condition(review):
    if not review.strip():
        return "Unknown"
    vectorized_input = tfidf_vectorizer.transform([review])
    predicted_condition_encoded = rf_classifier.predict(vectorized_input)[0]
    return label_encoder.inverse_transform([predicted_condition_encoded])[0]

def identify_key_elements(reviews):
    if not reviews:
        return "No significant elements"
    tfidf_vector = tfidf_vectorizer.transform(reviews)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    stop_words = set(stopwords.words("english"))
    meaningful_terms = np.array([word for word in feature_names if word.lower() not in stop_words])
    return ", ".join(meaningful_terms[np.argsort(tfidf_vector.toarray().sum(axis=0))[-5:]])

combined_reviews = df_filtered["review"].tolist()
predicted_conditions = [predict_condition(review) for review in combined_reviews]
most_common_condition = max(set(predicted_conditions), key=predicted_conditions.count) if predicted_conditions else "Unknown"

if selected_constraint == "Predict a patient’s condition based on reviews":
    user_review = st.text_area("Write a drug review here...")
    if st.button("Analyze Review"):
        if user_review.strip():
            user_pred = predict_condition(user_review)
            user_elements = identify_key_elements([user_review])
            user_sentiment = classify_sentiment(estimate_rating([user_review]))
            
            st.write(f"🧪 **User Predicted Condition:** {user_pred}")
            st.write(f"🔑 **User Key Elements:** {user_elements}")
            st.write(f"🎭 **User Sentiment:** {user_sentiment}")
    
elif selected_constraint == "Estimate drug ratings from reviews":
    avg_rating = df_filtered["rating"].mean()
    st.write(f"⭐ **Estimated Rating:** {avg_rating:.2f}")
elif selected_constraint == "Identify key elements that make reviews helpful":
    key_elements = identify_key_elements(combined_reviews)
    st.write(f"🔑 **Key Elements:** {key_elements}")
elif selected_constraint == "Classify reviews as Positive, Neutral, or Negative":
    avg_rating = df_filtered["rating"].mean()
    overall_sentiment = classify_sentiment(avg_rating)
    st.write(f"🎭 **Overall Sentiment:** {overall_sentiment}")
