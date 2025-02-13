import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the CSV data
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Streamlit app
st.title('HR Employee Attrition Analysis with NLP Features')

# Display the data
st.subheader('Employee Data')
st.write(data)

# Show basic statistics
st.subheader('Basic Statistics')
st.write(data.describe())

# Show attrition count
st.subheader('Attrition Count')
attrition_count = data['Attrition'].value_counts()
st.bar_chart(attrition_count)

# Show correlation heatmap
st.subheader('Correlation Heatmap')
correlation = data.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
st.image('correlation_heatmap.png')

# Generate word cloud for JobRole
st.subheader('Word Cloud for Job Roles')
job_roles = ' '.join(data['JobRole'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(job_roles)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud_job_roles.png')
st.image('wordcloud_job_roles.png')

# Dependencies versions
dependencies = """
pandas==1.3.3
streamlit==0.89.0
seaborn==0.11.2
matplotlib==3.4.3
wordcloud==1.8.1
"""
st.subheader('Dependencies Versions')
st.code(dependencies, language='text')
