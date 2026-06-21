import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    y = []

    for word in words:
        if word.isalnum() and word not in stop_words:
            y.append(ps.stem(word))

    return " ".join(y)

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")