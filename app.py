import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard")

@st.cache_data
def load_data():
    file_id = '1jm9pe3cVWE6LgF_YxDrxwg6a56oljQ2n'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'cleaned_data.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

df = load_data()

st.markdown("### Dataset Preview")
st.dataframe(df.head())

# Show Sentiment Distribution
st.markdown("### Sentiment Distribution")
sentiment_counts = df['Sentiment'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', ax=ax)
ax.set_ylabel("Count")
ax.set_xlabel("Sentiment")
st.pyplot(fig)

# Pie Chart
st.markdown("### Sentiment Distribution Pie Chart")
fig2, ax2 = plt.subplots()
ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis"))
st.pyplot(fig2)

# Word Count per Review
st.markdown("### Review Word Count Distribution")
df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
fig3, ax3 = plt.subplots()
sns.histplot(df['word_count'], bins=30, kde=True, ax=ax3)
ax3.set_xlabel("Word Count")
ax3.set_ylabel("Number of Reviews")
st.pyplot(fig3)
