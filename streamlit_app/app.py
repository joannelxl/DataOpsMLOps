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
import pyLDAvis
import pyLDAvis.gensim_models

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
    
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    return html_string

#Customer Sentiment Analysis Page
if selected_label == "Customer Sentiment Analysis":
    st.title('🏂 MBS Customer Sentiment Analysis') 

    # this is filter by date_input
    # min_date = complete_df['StayDate'].min()
    # max_date = complete_df['StayDate'].max()
    # date_range = st.date_input("Pick the period you want to analyse", (min_date, max_date),  max_value = max_date, min_value = min_date, format="DD/MM/YYYY")
    
    # this is filter by year only
    year_list = list(df.StayDateYear.unique())[::-1]
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
    df_selected_year = df[df.StayDateYear == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="Rating", ascending=False)


    #co11, col2 = st.columns([2,3])

    # either filter by year or filter by range of dates

    #filter by year
    filter_year = (df["StayDateYear"] == selected_year)
    filtered_df_by_year = df[filter_year]

    # filter by range of dates
    # earliest_date = pd.to_datetime(date_range[0])
    # latest_date = pd.to_datetime(date_range[1])

    # filtered_df_by_year = complete_df[(complete_df['StayDate'] >= earliest_date) & (complete_df['StayDate'] <= latest_date)]
    # st.write(filtered_df_by_year)

    #with co11:
    #st.write("""
            ### Bar Chart of Customer's Reviews""")
    # fig, ax = plt.subplots()
    rating_counts = filtered_df_by_year['Rating'].value_counts().sort_index()
    # ax.bar(counts.index, counts.values)
    # ax.set_xlabel('Ratings')
    # ax.set_ylabel('Count')
    # ax.set_xticks(range(1, 6))
    # ax.set_xticklabels(range(1, 6))
    # st.pyplot(fig)
    fig1 = px.bar(x = rating_counts.index, y = rating_counts.values, 
                  title= "Count of Customer Reviews", labels={"x": "Ratings", "y": "Count"})
    st.plotly_chart(fig1)

    #with col2: 
    # st.write("""
    #     ### Pie Chart of Positive and Negative Review Sentiments
    #             """)
    # fig, ax = plt.subplots()
    # filtered_df_by_year['Text_Sentiment'] = filtered_df_by_year['Text_Sentiment'].map({0: 'Negative', 1: 'Positive'})
    sentiment_counts = filtered_df_by_year['Text_Sentiment'].value_counts().reset_index()
    sentiment_counts["Text_Sentiment"] = sentiment_counts["Text_Sentiment"].replace({
        0 : "Negative",
        1 : "Positive"
    })
    # wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.1f%%', startangle=90)
    # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # ax.legend(wedges, counts.index, title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5), )
    # plt.setp(autotexts, size=8, weight="bold")  # Adjust the size and weight of percentage labels
    # st.pyplot(fig)
    fig2 = px.pie(sentiment_counts, values = "count", names = sentiment_counts["Text_Sentiment"],
                  title= "Proportion of Sentiment for Customer Reviews", 
                  color=sentiment_counts["Text_Sentiment"],
                  color_discrete_sequence=["sky blue", "crimson"])
    st.plotly_chart(fig2)

else: #selected_label = "Topic Modelling"
    st.title('🏂 Topic Modelling')
    
    #Obtain all the nouns from the text
    df = preprocess_data(df, "CleanReviewText")

    #Positive Reviews Topic Modelling
    positive_df = df[df["Text_Sentiment"] == 1]

    positive_html_string = topic_modelling(positive_df)
    components.v1.html(positive_html_string, width=1200, height=800, scrolling=True)

    #Negative Reviews Topic Modelling
    negative_df = df[df["Text_Sentiment"] == 0]

    negative_html_string = topic_modelling(negative_df)
    components.v1.html(negative_html_string, width=1200, height=800, scrolling=True)

