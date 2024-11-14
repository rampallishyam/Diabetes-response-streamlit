import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_table("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt")
    return df

df = load_data()

model = GradientBoostingRegressor(random_state=0)
model.fit(df.iloc[:,[0] + list(range(2, df.shape[1]-1))],df["Y"])

# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

st.sidebar.title("Input features")
age = st.sidebar.slider("Age",min(df["AGE"]),max(df["AGE"]))
bmi = st.sidebar.slider("Body Mass Index",min(df["BMI"]),max(df["BMI"]))
bp = st.sidebar.slider("Blood Pressure",min(df["BP"]),max(df["BP"]))
s1 = st.sidebar.slider("total serum cholesterol",min(df["S1"]),max(df["S1"]))
s2 = st.sidebar.slider("low-density lipoproteins",min(df["S2"]),max(df["S2"]))
s3 = st.sidebar.slider("high-density lipoproteins",min(df["S3"]),max(df["S3"]))
s4 = st.sidebar.slider("total cholesterol / HDL",min(df["S4"]),max(df["S4"]))
s5 = st.sidebar.slider("possibly log of serum triglycerides level",min(df["S5"]),max(df["S5"]))
s6 = st.sidebar.slider("blood sugar level",min(df["S6"]),max(df["S6"]))

st.text("The response variable, a measure of disease progression one year after baseline is ")
st.text(f"{model.predict(np.array([age,bmi,bp,s1,s2,s3,s4,s5,s6]).reshape(1, -1))}")

