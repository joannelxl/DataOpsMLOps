import streamlit as st
import pandas as pd
import mysql.connector
import sqlalchemy
from mysql.connector import Error
from datetime import date, timedelta
import yaml
from sqlalchemy import create_engine

@st.cache_data
def get_data():
    with open('db_info.yml', 'r') as file:
        data = yaml.safe_load(file)

        db_host = data["DATABASE_ENDPOINT"]
        db_user = data["DATABASE_USERNAME"]
        db_pw = data["DATABASE_PASSWORD"]
        db_name = data["DATABASE_NAME"]
        db_port = data["DATABASE_PORT"]

        dwh_host = data["DATAWH_ENDPOINT"]
        dwh_user = data["DATAWH_USERNAME"]
        dwh_pw = data["DATAWH_PASSWORD"]
        dwh_name = data["DATAWH_NAME"]
        dwh_port = data["DATAWH_PORT"]

        #engine = create_engine(f'mysql://admin:bt4301dw1@bt4301-datawarehouse1.cxmwooi4whw1.ap-southeast-1.rds.amazonaws.com:3306/BT4301_G09_DataWarehouse')
    try:
        engine = create_engine(f'mysql+pymysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False)
        review_sql = '''
        SELECT * 
        FROM review;'''
        review_df = pd.read_sql(review_sql, engine.raw_connection())

        time_sql = '''
        SELECT * 
        FROM time;'''

        time_df = pd.read_sql(time_sql, engine.raw_connection())

        fact_sql = '''
        SELECT * 
        FROM fact;'''

        fact_df = pd.read_sql(fact_sql, engine.raw_connection())

        join_sql = '''
        SELECT  fact.Text_Sentiment, fact.Title_Sentiment, time.OverallID, time.StayDate, time.StayDateYear, time.StayDateMonth, time.StayDateDayOfWeek, time.StayDateDay, time.StayDateWeek,
            review.ReviewID, review.CleanReviewTitle, review.CleanReviewText, review.AuthorContribution, review.Rating, review.WeightedTitleScore, review.WeightedTextScore
        FROM fact
        INNER JOIN review ON fact.OverallID = review.OverallID
        INNER JOIN time ON time.OverallID = fact.OverallID
        ORDER BY time.OverallID;
        '''


        complete_df = pd.read_sql(join_sql, engine.raw_connection())
    except Exception as e:
        st.write(e)
        st.error("Unable to connect to DB, please check error logs")
    return complete_df

