import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load trained pipeline
model_path = "emotion_pipeline.pkl"
with open(model_path, "rb") as f:
    pipeline = pickle.load(f)

# Emotion labels
emotion_labels = {
    0: "Anger 😠",
    1: "Fear 😨",
    2: "Joy 😄",
    3: "Love ❤️",
    4: "Sadness 😢",
    5: "Surprise 😲"
}

# App UI
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

st.title("🧠 Emotion Detection from Text")
st.write("This NLP app predicts **human emotions** from text using Machine Learning.")

# Text input
text = st.text_area("✍️ Enter your text here:")

# Predict button
if st.button("🔍 Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = pipeline.predict([text])[0]
        st.write(f"Prediction raw output: {prediction}")

        

# WordCloud
st.markdown("---")
st.subheader("☁️ Word Cloud")

if st.button("Generate Word Cloud"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        wc = WordCloud(
            width=600,
            height=300,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, NLP & Machine Learning")
