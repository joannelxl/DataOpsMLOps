import streamlit as st
import pandas as pd
import altair as alt
import sqlalchemy
import plotly.express as px
import matplotlib.pyplot as plt
import mysql.connector
from sqlalchemy import create_engine
import yaml
from db_utils import get_data
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from streamlit import components
from datetime import datetime
import pyLDAvis
import pyLDAvis.gensim_models
import re

# Set basic configurations for the Streamlit page
st.set_page_config(
    page_title="MBS Customer Sentiment Analysis",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

#Define the two pages for our app
pages = ["Customer Sentiment Analysis", "Topic Modelling"]
selected_label = st.sidebar.radio("Pages", pages)

#Load our data from dwh
df = get_data()

@st.cache_data
def preprocess_data(df, column):
    def preprocess_text(text):
        is_noun = lambda pos: pos[:2] == 'NN'
        text = text.split()
        nouns = [word for (word, pos) in pos_tag(text) if is_noun(pos)]
        return nouns
    df['ProcessedText'] = df[column].apply(preprocess_text)
    return df

@st.cache_data
def topic_modelling(df):
    #Apply topic modelling package
    texts = df["ProcessedText"].to_numpy()
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus,
                    id2word=dictionary,
                    num_topics=5,
                    random_state=42,
                    update_every=1,
                    chunksize=100,
                    passes=5,
                    alpha='auto',
                    per_word_topics=True)
    
    # Code referenced from https://stackoverflow.com/questions/46536132/how-to-access-topic-words-only-in-gensim
    print(lda_model.show_topics())
    lda_topics = lda_model.show_topics()
    words = {}
    for topic, word in lda_topics:
        words[topic] = re.sub('[^A-Za-z ]+', '', word)
    # End of code reference
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # Get topic mapping
    updated_mapping = {}
    for i in vis.topic_coordinates.topics.index:
        print(i)
        print(i%5)
        print(words[i])
        updated_mapping[vis.topic_coordinates.topics[i]] = words[i]
    print(updated_mapping)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    return (html_string, updated_mapping)

#Customer Sentiment Analysis Page
if selected_label == "Customer Sentiment Analysis":
    st.title('🏂 MBS Customer Sentiment Analysis') 

    # this is filter by date_input
    min_date = df['StayDate'].min()
    max_date = df['StayDate'].max()
    date_range = st.date_input("Pick the period you want to analyse", (min_date, max_date),  max_value = max_date, min_value = min_date, format="DD/MM/YYYY", key="sentiment")

    # filter by range of dates
    earliest_date = pd.to_datetime(date_range[0])
    latest_date = max_date if len(date_range) == 1 else pd.to_datetime(date_range[1]) # ensure that there will always be a valid end date

    filtered_df_by_year = df[(df['StayDate'] >= earliest_date) & (df['StayDate'] <= latest_date)]

    rating_counts = filtered_df_by_year['Rating'].value_counts().sort_index()

    if not rating_counts.empty:
        fig1 = px.bar(x = rating_counts.index, y = rating_counts.values, 
                    title= "Count of Customer Reviews", labels={"x": "Ratings", "y": "Count"})
        st.plotly_chart(fig1)
    else:
        st.markdown("<h2 style='text-align: center;'>No data available for the selected period.</h2>", unsafe_allow_html=True)

    sentiment_counts = filtered_df_by_year['Text_Sentiment'].value_counts().reset_index()
    sentiment_counts["Text_Sentiment"] = sentiment_counts["Text_Sentiment"].replace({
        0 : "Negative",
        1 : "Positive"
    })

    if not sentiment_counts.empty:
        fig2 = px.pie(sentiment_counts, values = "count", names = sentiment_counts["Text_Sentiment"],
                    title= "Proportion of Sentiment for Customer Reviews", 
                    color=sentiment_counts["Text_Sentiment"],
                    color_discrete_sequence=["sky blue", "crimson"])
        st.plotly_chart(fig2)

else: #selected_label = "Topic Modelling"
    st.title('🏂 Topic Modelling')

    # this is filter by date_input
    min_date = df['StayDate'].min()
    max_date = df['StayDate'].max()
    date_range = st.date_input("Pick the period you want to analyse", (min_date, max_date),  max_value = max_date, min_value = min_date, format="DD/MM/YYYY", key="topics")
    
    # filter by range of dates
    earliest_date = pd.to_datetime(date_range[0])
    latest_date = max_date if len(date_range) == 1 else pd.to_datetime(date_range[1]) # ensure that there will always be a valid end date
    filtered_df_by_year = df[(df['StayDate'] >= earliest_date) & (df['StayDate'] <= latest_date)]

    #Obtain all the nouns from the text
    df = preprocess_data(filtered_df_by_year, "CleanReviewText")

    #Positive Reviews Topic Modelling
    positive_df = df[df["Text_Sentiment"] == 1]

    st.markdown("<h1 style='text-align: center;'>Topics for Positive Reviews</h1>", unsafe_allow_html=True)
    if not positive_df.empty:
        positive_html_string, tokens = topic_modelling(positive_df)
        components.v1.html(positive_html_string, width=1200, height=800, scrolling=True)
        positive_tokens_df = pd.DataFrame(columns=["Topic", "Most Relevant Words"])
        for topic in tokens:
            positive_tokens_df.loc[len(positive_tokens_df.index)] = [topic, list(filter(lambda token: token != "", tokens[topic].split(" ")))]
        st.dataframe(positive_tokens_df, width=1200, hide_index=True)
    else:
        st.markdown("<h2 style='text-align: center;'>No data available for the selected period.</h2>", unsafe_allow_html=True)

    # Line break
    st.markdown("---")

    #Negative Reviews Topic Modelling
    negative_df = df[df["Text_Sentiment"] == 0]

    st.markdown("<h1 style='text-align: center;'>Topics for Negative Reviews</h1>", unsafe_allow_html=True)
    if not negative_df.empty:
        negative_html_string, tokens = topic_modelling(negative_df)
        components.v1.html(negative_html_string, width=1200, height=800, scrolling=True)
        negative_tokens_df = pd.DataFrame(columns=["Topic", "Most Relevant Words"])
        for topic in tokens:
            negative_tokens_df.loc[len(negative_tokens_df.index)] = [topic, list(filter(lambda token: token != "", tokens[topic].split(" ")))]
        st.dataframe(negative_tokens_df, width=1200, hide_index=True)
    else:
        st.markdown("<h2 style='text-align: center;'>No data available for the selected period.</h2>", unsafe_allow_html=True)