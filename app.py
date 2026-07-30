import streamlit as st
import joblib

# Load Models
classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")

# Load Scalers
classification_scaler = joblib.load("classification_scaler.pkl")
regression_scaler = joblib.load("regression_scaler.pkl")

st.set_page_config(page_title="Session 24 AIML Models")

st.title("Session 24 AIML Models")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Classification", "Regression"]
)

# ==========================
# CLASSIFICATION
# ==========================

if model_type == "Classification":

    st.header("Iris Flower Classification")

    sepal_length = st.number_input("Sepal Length", value=5.1)
    sepal_width = st.number_input("Sepal Width", value=3.5)
    petal_length = st.number_input("Petal Length", value=1.4)
    petal_width = st.number_input("Petal Width", value=0.2)

    if st.button("Predict Classification"):

        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        input_data = classification_scaler.transform(input_data)

        prediction = classification_model.predict(input_data)

        flower = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        st.success("Predicted Flower: " + flower[prediction[0]])
# ==========================
# REGRESSION
# ==========================

elif model_type == "Regression":

    st.header("Student Math Score Prediction")

    gender = st.number_input("Gender (0=Female, 1=Male)", min_value=0, max_value=1, value=0)
    race = st.number_input("Race/Ethnicity (0-4)", min_value=0, max_value=4, value=0)
    parent = st.number_input("Parental Education (0-5)", min_value=0, max_value=5, value=0)
    lunch = st.number_input("Lunch (0=Free/Reduced, 1=Standard)", min_value=0, max_value=1, value=1)
    prep = st.number_input("Test Preparation (0=None, 1=Completed)", min_value=0, max_value=1, value=0)
    reading = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
    writing = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

    if st.button("Predict Math Score"):

        input_data = [[
            gender,
            race,
            parent,
            lunch,
            prep,
            reading,
            writing
        ]]

        input_data = regression_scaler.transform(input_data)

        prediction = regression_model.predict(input_data)

        st.success("Predicted Math Score: " + str(round(prediction[0], 2)))
