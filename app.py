import streamlit as st
import joblib

model = joblib.load("iris_model.pkl")

st.title("ML Iris Model Prediction")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Make Prediction"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(flowers[pred[0]])