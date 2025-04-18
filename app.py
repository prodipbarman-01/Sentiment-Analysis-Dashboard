import streamlit as st
import pandas as pd

# Title of the app
st.title('Sentiment Analysis of Reviews')

# Load the cleaned data
def load_data():
    df = pd.read_csv('cleaned_data.csv')  # Ensure your cleaned data is named 'cleaned_data.csv'
    return df

df = load_data()

# Display the first few rows of the data
st.write("### First few rows of the data", df.head())

# Sentiment prediction (using a simple logic here; you can replace with your model)
def predict_sentiment(text):
    if 'bad' in text or 'poor' in text:
        return 'negative'
    elif 'good' in text or 'excellent' in text:
        return 'positive'
    else:
        return 'neutral'

# Apply sentiment prediction to review content
df['Predicted_Sentiment'] = df['Text'].apply(predict_sentiment)

# Display the data with the predicted sentiments
st.write("### Data with predicted sentiments", df[['Text', 'Predicted_Sentiment']])

# Display summary statistics
st.write("### Sentiment Distribution", df['Predicted_Sentiment'].value_counts())

# Allow user to input text for sentiment prediction
user_input = st.text_area("Enter review text for sentiment prediction:")
if user_input:
    prediction = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {prediction}")
