import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Spam Detection",
    page_icon="📩",
    layout="wide"
)

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return vectorizer, model

tfidf, model = load_model()

# -----------------------------
# NLP Objects
# -----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Session State
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Functions
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    processed = []

    for word in words:
        if word.isalnum() and word not in stop_words:
            processed.append(ps.stem(word))

    return " ".join(processed), processed


def predict_message(message):
    processed_text, tokens = preprocess_text(message)

    vector = tfidf.transform([processed_text])

    prediction = model.predict(vector)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vector).max() * 100

    return prediction, confidence, processed_text, tokens


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Module")

st.sidebar.markdown("""
### 🧠 Model Details

- **Algorithm:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF
- **NLP Techniques**
  - Lowercasing
  - Tokenization
  - Stopword Removal
  - Stemming

---

### 📚 Dataset

SMS Spam Collection Dataset

---

### 👨‍💻 Technologies

- Python
- Streamlit
- Scikit-learn
- NLTK
""")

# -----------------------------
# Title
# -----------------------------
st.title("📩 Email & SMS Spam Detection")

st.write(
    "This module classifies an Email or SMS as **Spam** or **Not Spam** "
    "using Machine Learning and Natural Language Processing."
)

# -----------------------------
# Examples
# -----------------------------
with st.expander("💡 Try Sample Messages"):

    st.code(
        "Congratulations! You have won a FREE iPhone. Click here to claim."
    )

    st.code(
        "Hey, are we meeting tomorrow at 6 PM?"
    )

# -----------------------------
# Input
# -----------------------------
input_sms = st.text_area(
    "Enter your Email or SMS",
    height=180,
    placeholder="Type your message here..."
)

# -----------------------------
# Predict
# -----------------------------
if st.button("🚀 Predict", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")

    else:

        prediction, confidence, processed_text, tokens = predict_message(input_sms)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📊 Message Statistics")

            st.metric("Characters", len(input_sms))

            st.metric("Words", len(input_sms.split()))

            st.metric("Processed Tokens", len(tokens))

        with col2:

            st.subheader("🎯 Prediction")

            if prediction == 1:
                st.error("🚨 SPAM MESSAGE")
            else:
                st.success("✅ NOT SPAM")

            if confidence is not None:
                st.progress(confidence / 100)
                st.write(f"**Confidence : {confidence:.2f}%**")

        st.divider()

        st.subheader("📝 Preprocessed Text")

        st.info(processed_text)

        st.subheader("🔤 Tokens")

        st.write(tokens)

        report = f"""
Spam Detection Report
---------------------

Date:
{datetime.now()}

Original Message:
{input_sms}

Processed Text:
{processed_text}

Prediction:
{"Spam" if prediction==1 else "Not Spam"}

Confidence:
{confidence:.2f}% if confidence else "N/A"
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="spam_report.txt"
        )

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": input_sms[:40],
            "prediction": "Spam" if prediction == 1 else "Not Spam"
        })

# -----------------------------
# History
# -----------------------------
if st.session_state.history:

    st.divider()

    st.subheader("🕘 Prediction History")

    for item in reversed(st.session_state.history):

        icon = "🚨" if item["prediction"] == "Spam" else "✅"

        st.write(
            f"{icon} **{item['prediction']}** | "
            f"{item['time']} | "
            f"{item['message']}..."
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Built using ❤️ Python | Streamlit | Scikit-learn | NLTK"
)