import streamlit as st
import pandas as pd
import altair as alt
import sqlalchemy
import plotly.express as px
import os
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
SELECT  time.OverallID, time.TimeID, time.StayDate, time.StayDateYear, time.StayDateMonth, time.StayDateDayOfWeek, time.StayDateDay, time.StayDateWeek,
    review.ReviewID, review.CleanReviewTitle, review.CleanReviewText, review.DateOfStay, review.AuthorContribution, review.Rating, review.TextBlob_Title, review.TextBlob_Review
FROM fact
INNER JOIN review ON fact.OverallID = review.OverallID
INNER JOIN time ON time.OverallID = fact.OverallID
ORDER BY time.OverallID
'''

complete_df = pd.read_sql(join_sql, engine)
st.write(complete_df)

with st.sidebar:
    st.title('🏂 MBS Customer Sentiment Analysis')    
    
    year_list = list(complete_df.StayDateYear.unique())[::-1]
    month_list = list(complete_df.StayDateMonth.unique())[::-1]
    
    selected_month = st.selectbox('Select a month', month_list, index = len(month_list) -1)
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
    df_selected_year = complete_df[complete_df.StayDateYear == selected_year]
    df_selected_month = complete_df[complete_df.StayDateMonth == selected_month]

    df_selected_year_sorted = df_selected_year.sort_values(by="Rating", ascending=False)
    df_selected_month_sorted = df_selected_month.sort_values(by="Rating", ascending=False)


    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)