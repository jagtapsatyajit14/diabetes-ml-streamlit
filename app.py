# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# LOAD DATA
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\Streamlit\diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# SPLIT FEATURES AND LABEL
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# USER INPUT FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL TRAINING
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALIZATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
color = 'blue' if user_result[0] == 0 else 'red'

# GRAPH FUNCTIONS
def draw_scatter(x_col, y_col, palette, user_x, user_y, title):
    fig = plt.figure()
    sns.scatterplot(x=x_col, y=y_col, data=df, hue='Outcome', palette=palette)
    sns.scatterplot(x=user_x, y=user_y, s=150, color=color)
    plt.title(title)
    st.pyplot(fig)

# Individual Graphs
st.header('Pregnancy count Graph (Others vs Yours)')
draw_scatter('Age', 'Pregnancies', 'Greens', user_data['Age'], user_data['Pregnancies'], '0 - Healthy & 1 - Unhealthy')

st.header('Glucose Value Graph (Others vs Yours)')
draw_scatter('Age', 'Glucose', 'magma', user_data['Age'], user_data['Glucose'], '0 - Healthy & 1 - Unhealthy')

st.header('Blood Pressure Value Graph (Others vs Yours)')
draw_scatter('Age', 'BloodPressure', 'Reds', user_data['Age'], user_data['BloodPressure'], '0 - Healthy & 1 - Unhealthy')

st.header('Skin Thickness Value Graph (Others vs Yours)')
draw_scatter('Age', 'SkinThickness', 'Blues', user_data['Age'], user_data['SkinThickness'], '0 - Healthy & 1 - Unhealthy')

st.header('Insulin Value Graph (Others vs Yours)')
draw_scatter('Age', 'Insulin', 'rocket', user_data['Age'], user_data['Insulin'], '0 - Healthy & 1 - Unhealthy')

st.header('BMI Value Graph (Others vs Yours)')
draw_scatter('Age', 'BMI', 'rainbow', user_data['Age'], user_data['BMI'], '0 - Healthy & 1 - Unhealthy')

st.header('DPF Value Graph (Others vs Yours)')
draw_scatter('Age', 'DiabetesPedigreeFunction', 'YlOrBr', user_data['Age'], user_data['DiabetesPedigreeFunction'], '0 - Healthy & 1 - Unhealthy')

# OUTPUT
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

# ACCURACY
st.subheader('Model Accuracy:')
st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")
