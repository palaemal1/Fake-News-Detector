import streamlit as st 
import joblib

vectorizer=joblib.load("./models/vectorizer.jb")
model=joblib.load("./models/lr_model.jb")

st.title("Fake News Detector")
st.write("Enter a News Article below to Check whether It Is Fake or Real.")

news_input=st.text_area("News article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input=vectorizer.transform([news_input])
        prediction=model.predict(transform_input)

        if prediction[0]==1:
            st.success("The news is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")


