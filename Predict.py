import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_education = data["le_education"]

def show_predict():
    st.title('Data Job Salary Prediction!')
    
    st.write("### We Need Some Information to Predict the Salary!")
    
    job = ('Data Engineering', 'Data Architecture and Modeling',
       'Data Science and Research', 'Machine Learning and AI',
       'Data Analysis', 'Leadership and Management',
       'BI and Visualization', 'Data Quality and Operations',
       'Data Management and Strategy', 'Cloud and Database')
    
    countries = (
        'United States',
        'United Kingdom',
        'Other',
        'Canada',
        'Spain',
        'Germany'
    )
    
    residence = (
        'Germany', 'United States', 'United Kingdom', 'Canada', 'Spain',
       'Ireland', 'South Africa', 'Poland', 'France', 'Netherlands',
       'Ukraine', 'Lithuania', 'Portugal', 'Australia', 'Uganda',
       'Colombia', 'Italy', 'Slovenia', 'Romania', 'Greece', 'India',
       'Latvia', 'Mauritius', 'Armenia', 'Croatia', 'Thailand',
       'South Korea', 'Estonia', 'Turkey', 'Philippines', 'Brazil',
       'Qatar', 'Russia', 'Kenya', 'Tunisia', 'Ghana', 'Belgium',
       'Switzerland', 'Andorra', 'Ecuador', 'Peru', 'Mexico', 'Moldova',
       'Nigeria', 'Saudi Arabia', 'Argentina', 'Egypt', 'Uzbekistan',
       'Georgia', 'Central African Republic', 'Finland', 'Austria',
       'Singapore', 'Sweden', 'Kuwait', 'Cyprus',
       'Bosnia and Herzegovina', 'Pakistan', 'Costa Rica', 'Chile',
       'Puerto Rico', 'Bolivia', 'Indonesia', 'United Arab Emirates',
       'Malaysia', 'Japan', 'Honduras', 'Czech Republic', 'Vietnam',
       'Iraq', 'Bulgaria', 'Jersey', 'Serbia', 'New Zealand', 'Hong Kong',
       'Denmark', 'Luxembourg', 'Malta'
    )
    
    exp_level=('Mid-level', 'Senior', 'Executive', 'Entry-level')
    
    company_sz = ('L', 'M', 'S')
    
    emp_type = 'Full-time'
    #dsf
    job_category = st.selectbox("Job Title", job)
    country = st.selectbox("Company Location", countries)
    experience = st.selectbox("Experience Level", exp_level)
    company_size = st.selectbox("Company Size", company_sz)
    emp_residence = st.selectbox("Employee Residence", residence)
    years_of_exp = st.slider("Years of Experience", 0 ,50, 3)
    
    ok = st.button("Predict The Salary")

    if ok:
        X = np.array([job_category, emp_residence, experience, emp_type, country, company_size])
        le_education = LabelEncoder()
        all_unique_values = np.unique(X)  
        le_education.fit(all_unique_values)

        X = le_education.transform(X)
        X = X.astype(float)
        Salary = regressor.predict(X.reshape(1, -1))
        st.subheader(f'The Estimated Salary is ${Salary[0]:.2f}')

    
    
    
    