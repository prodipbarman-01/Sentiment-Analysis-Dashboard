import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# Load the cleaned CSV
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_reviews.csv")  # Ensure this file is in the same folder
    return df

df = load_data()

# Title
st.title("🧠 Sentiment Analysis Dashboard")
st.markdown("Analyzing sentiments from product reviews")

# Show data preview
with st.expander("📄 Show Raw Data"):
    st.dataframe(df)

# Count of Sentiments
sentiment_counts = df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

# Pie chart
fig1 = px.pie(sentiment_counts, values='Count', names='Sentiment',
              title='Sentiment Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
st.plotly_chart(fig1, use_container_width=True)

# Bar chart
fig2 = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
              title='Sentiment Count Bar Chart', text='Count')
st.plotly_chart(fig2, use_container_width=True)

# Optional: Filter reviews by sentiment
selected_sentiment = st.selectbox("🔍 Filter by Sentiment", options=df['Sentiment'].unique())
filtered_reviews = df[df['Sentiment'] == selected_sentiment]

st.subheader(f"📝 Sample '{selected_sentiment}' Reviews")
st.write(filtered_reviews[['Text']].sample(min(5, len(filtered_reviews))))  # Show 5 or fewer

