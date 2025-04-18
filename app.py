import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown

# Set Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Download CSV from Google Drive
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # <-- এখানে তোমার Google Drive file ID বসাও
    output = "cleaned_data.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    df.columns = df.columns.str.strip()  # কলামের নাম থেকে স্পেস সরানো
    return df

# Load Data
df = load_data()

# Title
st.title("📊 Sentiment Analysis Dashboard")

# Show dataframe
with st.expander("🗃️ ডেটাসেট দেখুন"):
    st.dataframe(df.head(20))

# Sentiment Distribution
st.subheader("🧠 Sentiment Distribution")
if 'Sentiment' in df.columns:
    sentiment_counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="viridis")
    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.error("⚠️ 'Sentiment' নামের কোনো কলাম পাওয়া যায়নি!")

# Text Length Distribution (Optional)
if 'Text' in df.columns:
    df['Text Length'] = df['Text'].astype(str).apply(len)
    st.subheader("📝 টেক্সটের দৈর্ঘ্য")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Text Length'], bins=50, kde=True, ax=ax2, color="skyblue")
    ax2.set_title("Text Length Distribution")
    st.pyplot(fig2)
