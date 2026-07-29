import streamlit as st
import pandas as pd
import joblib

# Model load
pipe = joblib.load('full_pipeline.pkl')

st.title("📊 Student Math Score Predictor")

gender = st.selectbox('Gender', ['female', 'male'])
race = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
parent = st.selectbox('Parental Education', ["some high school", "high school", "associate's degree", "some college", "bachelor's degree", "master's degree"])
lunch = st.selectbox('Lunch', ['free/reduced', 'standard'])
test = st.selectbox('Test Preparation', ['none', 'completed'])
reading = st.slider('Reading Score', 0, 100, 72)
writing = st.slider('Writing Score', 0, 100, 74)

if st.button('Predict Math Score'):
    df = pd.DataFrame([{
        'gender': gender, 'race/ethnicity': race, 'parental level of education': parent,
        'lunch': lunch, 'test preparation course': test, 'reading score': reading, 'writing score': writing
    }])
    pred = pipe.predict(df)
    st.success(f"Predicted Math Score: {pred[0]:.2f}")
