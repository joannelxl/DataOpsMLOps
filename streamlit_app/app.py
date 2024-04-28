import streamlit as st
import pandas as pd
import sqlalchemy
import plotly.express as px
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
import plotly.graph_objects as go

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
def topic_modelling(df ,num_segments, top_tokens):
    #Apply topic modelling package
    texts = df["ProcessedText"].to_numpy()
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_segments,
                    random_state=42,
                    update_every=1,
                    chunksize=100,
                    passes=5,
                    alpha='auto',
                    per_word_topics=True)
    
    # Code referenced from https://stackoverflow.com/questions/46536132/how-to-access-topic-words-only-in-gensim
    lda_topics = lda_model.show_topics(num_words=top_tokens)
    words = {}
    for topic, word in lda_topics:
        words[topic] = re.sub('[^A-Za-z ]+', '', word)
    # End of code reference
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # Get topic mapping
    updated_mapping = {}
    for i in vis.topic_coordinates.topics.index:
        updated_mapping[vis.topic_coordinates.topics[i]] = words[i]
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
    
    month_counts = filtered_df_by_year["StayDate"].value_counts().reset_index()
    month_counts = month_counts.sort_values("StayDate", ascending=True).reset_index()
   
    month_counts["cumulative"] = month_counts["count"].cumsum()
    if not month_counts.empty:
        df = month_counts
        fig3 = go.Figure(data=go.Scatter(
            x=df["StayDate"], y= df["count"], name="Monthly"
        ))
        fig3.add_scatter(x=df["StayDate"], y=df["cumulative"], name="Cumulative")
        fig3.update_layout(title="Reviews Monthly and Cumulative",
                        xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig3)

    pos_df = filtered_df_by_year[filtered_df_by_year["Text_Sentiment"] == 1]
    
    neg_df = filtered_df_by_year[filtered_df_by_year["Text_Sentiment"] == 0]
    

    pos_counts = pos_df["StayDate"].value_counts().reset_index()
    pos_counts = pos_counts.sort_values("StayDate", ascending=True).reset_index()
    neg_counts = neg_df["StayDate"].value_counts().reset_index()
    neg_counts = neg_counts.sort_values("StayDate", ascending=True).reset_index()

    if not pos_counts.empty and not neg_counts.empty:
        fig4 = go.Figure(data=go.Scatter(
            x=pos_counts["StayDate"], y= pos_counts["count"], name="Positive Counts"
        ))
        fig4.add_scatter(x=neg_counts["StayDate"], y=neg_counts["count"], name="Negative Counts")
        fig4.data[1].line.color = "#FF0000" # Red line for negative reviews
        fig4.update_layout(title="Reviews Positive and Negative Counts",
                        xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig4)
        

else: #selected_label = "Topic Modelling"
    st.title('🏂 Topic Modelling')

    # this is filter by date_input
    min_date = df['StayDate'].min()
    max_date = df['StayDate'].max()
    date_range = st.date_input("Pick the period you want to analyse", (min_date, max_date),  max_value = max_date, min_value = min_date, format="DD/MM/YYYY", key="topics")

    # Split the view into 2 halves of equal width
    left, _, right = st.columns([0.4, 0.2, 0.4])
    with left:
        # slider to choose the number of segments to split the customers into
        selected_num_of_groups = st.slider("Number of customer segments",
                                           min_value=2, max_value=10, value=5)
    with right:
        # slider to choose the top relevant words associated to each customer segment
        selected_top_k_tokens_for_each_segment = st.slider("Number of most relevant words for each customer segment",
                                                           min_value=1, max_value=10, value=10)
    
    # filter by range of dates
    earliest_date = pd.to_datetime(date_range[0])
    latest_date = max_date if len(date_range) == 1 else pd.to_datetime(date_range[1]) # ensure that there will always be a valid end date
    filtered_df_by_year = df[(df['StayDate'] >= earliest_date) & (df['StayDate'] <= latest_date)]

    #Obtain all the nouns from the text
    df = preprocess_data(filtered_df_by_year, "CleanReviewText")

    #Positive Reviews Topic Modelling
    positive_df = df[df["Text_Sentiment"] == 1]

    st.markdown("<h1 style='text-align: center;'>Positive Reviews</h1>", unsafe_allow_html=True)
    if not positive_df.empty:
        positive_html_string, tokens = topic_modelling(positive_df, selected_num_of_groups, selected_top_k_tokens_for_each_segment)
        # components.v1.html(positive_html_string, width=1200, height=800, scrolling=True)
        positive_tokens_df = pd.DataFrame(columns=["Customer Segment", "Most Relevant Words"])
        for topic in tokens:
            positive_tokens_df.loc[len(positive_tokens_df.index)] = [topic, list(filter(lambda token: token != "", tokens[topic].split(" ")))]
        st.dataframe(positive_tokens_df, width=1200, hide_index=True)
        token1_clean = re.sub(r'\s+', ', ', tokens[1])
        st.write("The most relevant words associated with each customer segment tell us the areas where customers are satisfied with MBS hotel.")
        st.write("For example, customers from customer segment 1 are satisfied with: ",token1_clean)

    else:
        st.markdown("<h2 style='text-align: center;'>No data available for the selected period.</h2>", unsafe_allow_html=True)

    # Line break
    st.markdown("---")

    #Negative Reviews Topic Modelling
    negative_df = df[df["Text_Sentiment"] == 0]

    st.markdown("<h1 style='text-align: center;'>Negative Reviews</h1>", unsafe_allow_html=True)
    if not negative_df.empty:
        negative_html_string, tokens = topic_modelling(negative_df, selected_num_of_groups, selected_top_k_tokens_for_each_segment)
        # components.v1.html(negative_html_string, width=1200, height=800, scrolling=True)
        negative_tokens_df = pd.DataFrame(columns=["Customer Segment", "Most Relevant Words"])
        for topic in tokens:
            negative_tokens_df.loc[len(negative_tokens_df.index)] = [topic, list(filter(lambda token: token != "", tokens[topic].split(" ")))]
        st.dataframe(negative_tokens_df, width=1200, hide_index=True)
        token1_clean = re.sub(r'\s+', ', ', tokens[1])
        st.write("The most relevant words associated with each customer segment tell us the areas where customers are NOT satisfied with MBS hotel.")
        st.write("For example, customers from customer segment 1 are NOT satisfied with: ",token1_clean)
    else:
        st.markdown("<h2 style='text-align: center;'>No data available for the selected period.</h2>", unsafe_allow_html=True)