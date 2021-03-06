import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import brier_score_loss, make_scorer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

#define the Brier Skill Score scorer
brier_ref = 0.04632016076094936
def brier_skill_score(y_true, y_predicted):
    # calculate the brier score
    bs = brier_score_loss(y_true, y_predicted)
    # calculate skill score
    return 1.0 - (bs / brier_ref)

BSS_scorer = make_scorer(brier_skill_score, greater_is_better=True, needs_proba=True)


st.write("""
# Stroke Prediction App

This app predicts the probability of the onset of stroke!

Data obtained from the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle by fedesoriano.

Prediction powered by SupportVectorClassifier on sklearn.

Code can be found on [my Github repo](https://github.com/JoshJingtianWang/Stroke_Prediction).
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](./example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('Gender',('Male','Female'))
        age = st.sidebar.slider('Age', 0,130,30)
        hypertension = st.sidebar.selectbox('Hypertension',(0, 1))
        heart_disease = st.sidebar.selectbox('Heart Disease',(0, 1))
        ever_married = st.sidebar.selectbox('Ever married',('Yes', 'No'))
        work_type = st.sidebar.selectbox('Work Type',('Private', 'Self-employed', 'Govt_job', 'children'))
        Residence_type = st.sidebar.selectbox('Residence Type',('Rural', 'Urban'))
        avg_glucose_level = st.sidebar.slider('Average Glucose Level', 40,300,100)
        bmi = st.sidebar.slider('BMI', 8,80,25)
        smoking = st.sidebar.selectbox('Smoking Status',('never smoked', 'formerly smoked', 'smokes'))
        
        data = {'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': Residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking': smoking}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

df = input_df

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
#penguins_raw = pd.read_csv('penguins_cleaned.csv')
#penguins = penguins_raw.drop(columns=['species'])
#df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#encode = ['sex','island']
#for col in encode:
#    dummy = pd.get_dummies(df[col], prefix=col)
#    df = pd.concat([df,dummy], axis=1)
#    del df[col]
#df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_svc = pickle.load(open('stroke_gs_svc.pkl', 'rb'))

# Apply model to make predictions
prediction = load_svc.predict(df)
prediction_proba = load_svc.predict_proba(df)[0][1]


#st.subheader('Prediction')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
#st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write('''The probability of stroke onset is 

# %.3f'''%prediction_proba)


st.write('See where your probability lands among the test data:')

# Reads in saved X test proba
X_test_proba = pickle.load(open('X_test_proba.pkl', 'rb'))
percentile=percentileofscore(X_test_proba, prediction_proba)
#plotting the result in histogram
fig = plt.figure(figsize=(10, 4))
sns.histplot(data=X_test_proba).set(title='Test Data Probability Distribution')
#plot where the patient proba lands
plt.vlines(prediction_proba, 0, 500, color='r', label='', colors="r")
plt.annotate(str(round(percentile, 2))+'% percentile', xy=(prediction_proba+0.02, 300), weight='bold', color='r',
             xytext=(prediction_proba+0.08, 400), fontsize=15, arrowprops=dict(arrowstyle="->", color='r'))
plt.xlabel("Probability")
plt.xlim(0, 0.8)
#p.set_ylabel("Y-Axis", fontsize = 20)
st.pyplot(fig)