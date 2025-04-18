import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Function to download and load data from Google Drive
@st.cache_data
def load_data():
    file_id = '159TLOlDXH2Alvu4SgNG5iQiXUSSq2EdZ'
    url = f'https://drive.google.com/uc?id={file_id}'
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(BytesIO(response.content))
        return df
    else:
        st.error("⚠️ ফাইল ডাউনলোড করতে সমস্যা হয়েছে। দয়া করে লিঙ্ক চেক করুন।")
        return pd.DataFrame()

# Load data
df = load_data()

# Page title
st.title("📊 Sentiment Analysis Dashboard")
st.write("এই ড্যাশবোর্ডে প্রোডাক্ট রিভিউয়ের উপর ভিত্তি করে Sentiment (positive, neutral, negative) বিশ্লেষণ করা হয়েছে।")

# Show data
st.subheader("🔍 Raw Data Preview")
st.write(df.head())

# Show sentiment counts
st.subheader("📈 Sentiment Count")
sentiment_counts = df['Sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# Pie chart
st.subheader("🧁 Sentiment Distribution (Pie Chart)")
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue', 'lightgreen'])
ax.axis('equal')
st.pyplot(fig)

# Optional filter
st.subheader("🔎 Filtered Reviews by Sentiment")
option = st.selectbox("একটি sentiment নির্বাচন করুন:", df['Sentiment'].unique())
filtered_df = df[df['Sentiment'] == option]
st.write(filtered_df[['Text', 'Sentiment']].reset_index(drop=True))
