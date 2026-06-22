import streamlit as st
import pandas as pd
import requests
import datetime
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="NYT Analytics Portal", layout="wide")
st.title("NYT Headline Analysis & Word Prediction")

@st.cache_resource
def download_nltk_dependencies():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_dependencies()

def fetch_previous_month_headlines(api_key):
    today = datetime.date.today()
    first_of_this_month = today.replace(day=1)
    last_month_date = first_of_this_month - datetime.timedelta(days=1)
    
    year = last_month_date.year
    month = last_month_date.month
    
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        docs = data['response']['docs']
        
        headlines = []
        dates = []
        
        for doc in docs:
            if 'headline' in doc and 'main' in doc['headline']:
                headlines.append(doc['headline']['main'])
                dates.append(doc.get('pub_date', None))
                
        headlines_df = pd.DataFrame({'date': dates, 'headline': headlines})
        headlines_df['date'] = pd.to_datetime(headlines_df['date'], format='mixed', utc=True).dt.date
        return headlines_df
    else:
        raise Exception(f"API Request Failed: Status {response.status_code}")
    
def train_markov_chain(tokenized_headlines):
    model = defaultdict(Counter)
    for i in range(len(tokenized_headlines) - 1):
        current_word = tokenized_headlines[i]
        next_word = tokenized_headlines[i+1]
        model[current_word][next_word] += 1
    return model

def predict_next_words(model, starting_word, top_n=5):
    if starting_word not in model:
        return f"The word '{starting_word}' was not found in last month's text context."
    return model[starting_word].most_common(top_n)

st.sidebar.header("API & Parameter Controls")
api_key = st.sidebar.text_input("Enter NYT API Key", type="password")
lookback_months = st.sidebar.slider("Historical Window (Months)", 1, 6, 3)

@st.cache_data(ttl=86400) 
def load_and_preprocess_data(api_key, months):
    if not api_key:
        return pd.DataFrame()
    try:
        headlines_df = fetch_previous_month_headlines(api_key)
        return headlines_df
    except Exception as e:
        st.error(f"Error fetching real data from NYT: {e}")
        return pd.DataFrame()
    
if api_key:
    with st.spinner("Fetching data from New York Times systems..."):
        raw_data = load_and_preprocess_data(api_key, lookback_months)
        
    if not raw_data.empty:
        stop_words = set(stopwords.words('english'))
        all_sentences = raw_data['headline'].str.cat(sep=' ').lower()
        tokens = word_tokenize(all_sentences)
        cleaned_tokens = [
            word for word in tokens 
            if word not in string.punctuation and word not in stop_words and len(word) >= 3
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Contextual Word Cloud")
            wc = WordCloud(max_words=100, background_color="white", width=800, height=400).generate(' '.join(cleaned_tokens))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Primary Word Frequency Distribution")
            freq_dist = nltk.FreqDist(cleaned_tokens)
            most_common_df = pd.DataFrame(freq_dist.most_common(15), columns=['Word', 'Count'])
            
            fig, ax = plt.subplots()
            sns.barplot(data=most_common_df, x='Count', y='Word', palette='viridis', ax=ax)
            st.pyplot(fig)
            
        st.subheader("Daily Keyword Timeline & 30-Day Trend Forecast")
        target_word = st.text_input("Track Keyword Frequency over Time:", value="trump").lower()
        
        raw_data['day_string'] = raw_data['date'].astype(str)
        unique_days = sorted(raw_data['day_string'].unique())
        daily_counts = []
        
        for day in unique_days:
            day_text = raw_data[raw_data['day_string'] == day]['headline'].str.cat(sep=' ').lower()
            daily_counts.append(day_text.count(target_word))
            
        timeline_df = pd.DataFrame({'Day': unique_days, 'Occurrences': daily_counts})
        timeline_df['Type'] = 'Historical'
        
        # --- MACHINE LEARNING FORECASTING PIPELINE (OPTIMIZED AXIS) ---
        if len(timeline_df) > 1:
            X = np.array(range(len(timeline_df))).reshape(-1, 1)
            y = timeline_df['Occurrences'].values
            
            model_lr = LinearRegression()
            model_lr.fit(X, y)
            
            last_day_dt = pd.to_datetime(unique_days[-1])
            future_days = []
            future_predictions = []
            
            for i in range(1, 31):
                next_day = last_day_dt + datetime.timedelta(days=i)
                future_days.append(next_day.strftime('%Y-%m-%d'))
                
                pred_index = len(timeline_df) + i - 1
                pred_val = model_lr.predict([[pred_index]])[0]
                future_predictions.append(max(0, round(pred_val, 1)))
                
            forecast_df = pd.DataFrame({'Day': future_days, 'Occurrences': future_predictions})
            forecast_df['Type'] = 'Forecast'
            
            combined_df = pd.concat([timeline_df, forecast_df], ignore_index=True)
            
            fig_time, ax_time = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=combined_df, x='Day', y='Occurrences', hue='Type', palette=['#1f77b4', '#d62728'], ax=ax_time)
            
            # Fixes the crowded date axis by spacing out labels every 5 days
            import matplotlib.dates as mdates
            ax_time.set_xticks(combined_df['Day'][::5])
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_time)
        else:
            st.line_chart(data=timeline_df, x='Day', y='Occurrences')
        
        st.subheader("Predictive Language Forecast")
        st.write("Based on statistical adjacency distributions calculated from the text data pool:")
        
        markov_model = train_markov_chain(cleaned_tokens)
        
        only_words = [word for word in cleaned_tokens if word.isalpha()]
        top_words_keys = [item[0] for item in nltk.FreqDist(only_words).most_common(100)]
        
        seed_word = st.selectbox("Select a foundational root word to predict the next:", options=top_words_keys)
        
        predictions = predict_next_words(markov_model, seed_word)
        
        if isinstance(predictions, list) and len(predictions) > 0:
            pred_df = pd.DataFrame(predictions, columns=['Predicted Next Word', 'Observed Adjacency Weight'])
            pred_df_sorted = pred_df.sort_values(by='Observed Adjacency Weight', ascending=False)
            
            st.bar_chart(
                data=pred_df_sorted, 
                x='Predicted Next Word', 
                y='Observed Adjacency Weight',
                color="#4B0082"
            )
            
            with st.expander("Show detailed data table"):
                st.dataframe(pred_df, use_container_width=True)
        else:
            st.warning(f"No statistical successors found following the word '{seed_word}' in this specific month's context.")
            
else:
    st.info("Please provide a verified New York Times API credential key in the sidebar menu to launch analytics cascades.")