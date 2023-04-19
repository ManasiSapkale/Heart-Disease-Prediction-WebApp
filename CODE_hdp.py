import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


heart_data =pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\PROJECT\\Heart disease prediction project\\Heart.csv')



#loading the saved models

Heart_Disease =pickle.load(open('C:\\Users\\HP\\OneDrive\\Desktop\\PROJECT\\Heart disease prediction project\\Heart_model.sav','rb'))


#sidebar for navigate

with st.sidebar:
    
    selected = option_menu('PREDICTION_SYSTEM',
                           ['Heart Disease Prediction'],
                           
                           icons =['heart'],
                           
                           default_index =0)

   
if (selected == 'Heart Disease Prediction'):
    
    #page title
    st.title('*# HEART DISEASE PREDICTION #*')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
         age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain Types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestrol in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
        
    with col3:
        exang = st.text_input('Exercise Include Angina')
        
    with col1:
        oldpeak = st.text_input('ST Depression induced by Exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak Exercise ST segement')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect;2 = reversable defect')
        


# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(heart_data.drop(columns=['target']), heart_data['target'], test_size=0.2, random_state=42)

# Initialize the model
clf = RandomForestClassifier()

Heart_Disease.fit(X_train, Y_train)
predictions = Heart_Disease.predict(X_test)


# code for prediction
heart_diagnosis = ''

#creating a button for prediction

if st.button('Heart Disease Test Result'):
  
    predict = Heart_Disease.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    
    if (predict[0] == 1):
        heart_diagnosis = 'The person is having Heart Disease'
        
    else:
         heart_diagnosis = 'The person does not have any Heart Disease'


         
st.success(heart_diagnosis)
