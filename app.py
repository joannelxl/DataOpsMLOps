import streamlit as st
import pandas as pd
import altair as alt
import sqlalchemy
import plotly.express as px
import os
import matplotlib.pyplot as plt
import mysql.connector
from sqlalchemy import create_engine

st.set_page_config(
    page_title="MBS Customer Sentiment Analysis",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# I cant seem to use the env variable to connect to engine, but hard coding it works 
# dwh_host = os.getenv("DATAWH_ENDPOINT")
# dwh_user = os.getenv("DATAWH_USERNAME")
# dwh_pw = os.getenv("DATAWH_PASSWORD")
# dwh_name = os.getenv("DATAWH_NAME")
# dwh_port = os.getenv("DATAWH_PORT")
# engine = create_engine(f'mysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}')

engine = create_engine(f'mysql://admin:bt4301dw1@bt4301-datawarehouse1.cxmwooi4whw1.ap-southeast-1.rds.amazonaws.com:3306/BT4301_G09_DataWarehouse')

review_sql = '''
SELECT * 
FROM review'''
review_df = pd.read_sql(review_sql, engine)

time_sql = '''
SELECT * 
FROM time'''

time_df = pd.read_sql(time_sql, engine)

fact_sql = '''
SELECT * 
FROM fact'''

fact_df = pd.read_sql(fact_sql, engine)

join_sql = '''
SELECT  fact.Text_Sentiment, fact.Title_Sentiment, time.OverallID, time.StayDate, time.StayDateYear, time.StayDateMonth, time.StayDateDayOfWeek, time.StayDateDay, time.StayDateWeek,
    review.ReviewID, review.CleanReviewTitle, review.CleanReviewText, review.DateOfStay, review.AuthorContribution, review.Rating, review.WeightedTitleScore, review.WeightedTextScore
FROM fact
INNER JOIN review ON fact.OverallID = review.OverallID
INNER JOIN time ON time.OverallID = fact.OverallID
ORDER BY time.OverallID
'''


complete_df = pd.read_sql(join_sql, engine)
# st.write(complete_df)

with st.sidebar:
    st.title('🏂 MBS Customer Sentiment Analysis') 

    # this is filter by date_input
    # min_date = complete_df['StayDate'].min()
    # max_date = complete_df['StayDate'].max()
    # date_range = st.date_input("Pick the period you want to analyse", (min_date, max_date),  max_value = max_date, min_value = min_date, format="DD/MM/YYYY")
    
    # this is filter by year only
    year_list = list(complete_df.StayDateYear.unique())[::-1]
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
    df_selected_year = complete_df[complete_df.StayDateYear == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="Rating", ascending=False)


co11, col2 = st.columns([2,3])

# either filter by year or filter by range of dates

#filter by year
filter_year = (complete_df["StayDateYear"] == selected_year)
filtered_df_by_year = complete_df[filter_year]

# filter by range of dates
# earliest_date = pd.to_datetime(date_range[0])
# latest_date = pd.to_datetime(date_range[1])

# filtered_df_by_year = complete_df[(complete_df['StayDate'] >= earliest_date) & (complete_df['StayDate'] <= latest_date)]
# st.write(filtered_df_by_year)

with co11:
    st.write("""
        ### Bar Chart of Customer's Reviews""")
    fig, ax = plt.subplots()
    counts = filtered_df_by_year['Rating'].value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel('Ratings')
    ax.set_ylabel('Count')
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(range(1, 6))
    st.pyplot(fig)

with col2: 
    st.write("""
    ### Pie Chart of Positive and Negative Review Sentiments
             """)
    fig, ax = plt.subplots()
    filtered_df_by_year['Text_Sentiment'] = filtered_df_by_year['Text_Sentiment'].map({0: 'Negative', 1: 'Positive'})
    counts = filtered_df_by_year['Text_Sentiment'].value_counts()
    wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.legend(wedges, counts.index, title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5), )
    plt.setp(autotexts, size=8, weight="bold")  # Adjust the size and weight of percentage labels
    st.pyplot(fig)


