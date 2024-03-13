import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


def find_outliers(df, column_name, location_value):
    subset = df[df['company_location'] == location_value][column_name]
    
    Q1 = subset.quantile(0.25)
    Q3 = subset.quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df[(df['company_location'] == location_value) & ((df[column_name] < Q1 - 1.5 * IQR) | (df[column_name] > Q3 + 1.5 * IQR))]
    
    return outliers



def shorten_category(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def load_data():
    df = pd.read_csv(r"E:\Full Data Science Projects\Data Jobs Salary Prediction\dataset\jobs_in_data.csv")
    df['job_category'].replace({'Data Management and Strategy': 'Leadership and Management'}, inplace=True)
    df = df[['job_title' , 'job_category' , 'salary_in_usd' , 'employee_residence' , 'experience_level' , 'employment_type' , 'company_location' , 'company_size']]
    df = df[df['employment_type'] == 'Full-time']
    
    country_map =shorten_category(df.company_location.value_counts() , 60)
    df['company_location'] = df['company_location'].map(country_map)
    outliers_us_salary = find_outliers(df, 'salary_in_usd', 'United States')
    df.drop(outliers_us_salary.index, inplace=True)
    outliers_us_salary = find_outliers(df, 'salary_in_usd', 'Other')
    df.drop(outliers_us_salary.index, inplace=True)
    outliers_us_salary = find_outliers(df, 'salary_in_usd', 'United Kingdom')
    df.drop(outliers_us_salary.index, inplace=True)
    return df

df = load_data()



def expolre_page():
    # Plot 1: Job Categories Pie Chart (converted to Plotly)
    st.subheader("1. Job Categories Distribution - Pie Chart with Shadows (Top 5 Slices)")
    job_category_counts = df['job_category'].value_counts()
    labels = job_category_counts.index
    values = job_category_counts.values

    # Select top 5 slices
    top_5_labels = labels[:5]
    top_5_values = values[:5]
    other_label = 'Other'
    other_value = sum(values[5:])

    # Create a pie chart using Plotly with a different color sequence
    fig = px.pie(names=top_5_labels + [other_label],
                    values=top_5_values + [other_value],
                    hole=0.3,  # To create a doughnut-style pie chart
                    title="Job Categories Distribution",
                    labels={'label': 'Job Category', 'value': 'Count'},
                    color_discrete_sequence=px.colors.qualitative.Set3)

    # Display the pie chart using Streamlit
    st.plotly_chart(fig)
   
   
   # Plot 2
    st.subheader("2. Distribution of Salary")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['salary_in_usd'], bins=30, kde=True, ax=ax)
    plt.xlabel('Salary (USD)')
    plt.ylabel('Frequency')
    st.pyplot(fig)



    # Plot 3: Salary vs. Experience Level
    #
    st.subheader("3. Salary vs. Experience Level")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='experience_level', y='salary_in_usd', data=df, ax=ax)
    plt.xlabel('Experience Level')
    plt.ylabel('Salary (USD)')
    st.pyplot(fig)


    # Plot 4: Salary Distribution by Job Category
    st.subheader("4. Salary Distribution by Job Category")
    fig, ax = plt.subplots(figsize=(12, 6)) 
    sns.violinplot(x='job_category', y='salary_in_usd', data=df, ax=ax)
    plt.xlabel('Job Category')
    plt.ylabel('Salary (USD)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
