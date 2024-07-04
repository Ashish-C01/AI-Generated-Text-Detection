import streamlit as st
import tensorflow as tf
import pickle

with open('models/svm.pkl', 'rb') as f:
    svm = pickle.load(f)
with open('models/gnb.pkl', 'rb') as f:
    gnb = pickle.load(f)
with open('models/vclf.pkl', 'rb') as f:
    vclf = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)
model=tf.keras.models.load_model('models/ANN.h5')

option = st.selectbox(
    "Select the model",
    ("Naive Bayes", "Support Vector Machine", "Voting Classifier","ANN model"))

st.title("Detect AI generated text")
st.image("images.png")
user_input = st.text_area("Enter or paste the text here")
if st.button("Predict"):
    user_input = user_input.strip()
    if user_input != '':
        vectorized_text=tfidf.transform([user_input]).toarray()
        match option:
            case "Naive Bayes":
                prediction=gnb.predict(vectorized_text)
            case "Support Vector Machine":
                prediction=svm.predict(vectorized_text)
            case "Voting Classifier":
                prediction=vclf.predict(vectorized_text)
            case "ANN model":
                temp_result=model.predict(vectorized_text)
                prediction=1 if temp_result>0.5 else 0
        output="AI generated data" if prediction else "not an AI generated data"
        st.write(f"The text is predicted as {output}")
    else:
        st.warning("Please enter text to be predicted")


