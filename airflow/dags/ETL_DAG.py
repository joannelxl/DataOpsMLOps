from datetime import datetime, timedelta
import mysql.connector
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from uuid import uuid4
import numpy as np
from datetime import datetime
import re
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('vader_lexicon')
from textblob import TextBlob

import warnings
warnings.filterwarnings("ignore")
load_dotenv()

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

def process_text(text):
    # Initialise
    lemmatizer = WordNetLemmatizer()
    processed_text = " "
    
    # Process input
    text_lower = text.lower()
    word = word_tokenize(text_lower)
    
    # Alphabetical Tokens
    alphabetic_tokens = [word for word in word if re.match('^[a-zA-Z]+$', word)]
    
    # Remove stopwords from text and lemmatize
    stop_words = set(stopwords.words('english'))
    
    lem_words = []
    for word in alphabetic_tokens:
        if word not in stop_words:
            lem_words.append(lemmatizer.lemmatize(word))
    
    # Join the list of words
    processed_text = processed_text.join(lem_words)     #print(edited_stop_words)

    return processed_text

def clean_text(text):
    cleaned_text = ''.join([char.lower() for char in text if char.isalpha() or char.isspace()])
    return cleaned_text

def scale_reviews(value):
    # Calculate the scaled value using linear scaling
    scaled_value = ((value - 1) / (5 - 1)) * (1 - (-1)) + (-1)
    return scaled_value

def weighed_title_score(row):
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(row['CleanReviewTitle'])['compound']
    textblob = TextBlob(row['CleanReviewTitle']).sentiment.polarity
    rating = scale_reviews(row["Rating"])
    avg = (vader + textblob + rating) / 3
    return avg

def weighed_text_score(row):
    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(row['CleanReviewText'])['compound']
    textblob = TextBlob(row['CleanReviewText']).sentiment.polarity
    rating = scale_reviews(row["Rating"])
    avg = (vader + textblob + rating) / 3
    return avg

def threshold(value):
    if value < 0:
        return 0
    else:
        return 1

def set_overall_id(**kwargs):
    # create connection
    db_host = os.getenv("DATABASE_ENDPOINT")
    db_user = os.getenv("DATABASE_USERNAME")
    db_pw = os.getenv("DATABASE_PASSWORD")
    db_name = os.getenv("DATABASE_NAME")

    db_tripadvisor = mysql.connector.connect(
	host=db_host,
	user=db_user,
	passwd=db_pw,
	database=db_name
    )

    # Check if the OverallID column exists
    cursor = db_tripadvisor.cursor()
    cursor.execute("SHOW COLUMNS FROM tripadvisor_reviews LIKE 'OverallID'")
    if cursor.fetchone() is None:
        # If OverallID column doesn't exist, add it
        logging.info("fetchone")
        cursor.execute('ALTER TABLE tripadvisor_reviews ADD OverallID INT AUTO_INCREMENT PRIMARY KEY') 

def check_dwh_count(**kwargs):
    dwh_host = os.getenv("DATAWH_ENDPOINT")
    dwh_user = os.getenv("DATAWH_USERNAME")
    dwh_pw = os.getenv("DATAWH_PASSWORD")
    dwh_name = os.getenv("DATAWH_NAME")
    dwh_port = os.getenv("DATAWH_PORT")

    db_datawarehouse = mysql.connector.connect(
        host=dwh_host,
        user=dwh_user,
        passwd=dwh_pw,
        database=dwh_name,
        auth_plugin=dwh_pw
    )

    # check if table exist
    table = 'review' 
    cursor = db_datawarehouse.cursor()
    cursor.execute(f"SHOW TABLES LIKE '{table}'")
    table_exists = cursor.fetchone() is not None

    if table_exists == False:
        count = 0
        kwargs['ti'].xcom_push(key='count', value=0)
        return {'count':0}
    else:
        # Check for number of rows in dwh
        str_sql = '''
        SELECT COUNT(*)
        FROM review 
        '''
        df = pd.read_sql(sql=str_sql, con=db_datawarehouse)
        count = df.iloc[0,0]
        kwargs['ti'].xcom_push(key='count', value=count)
        logging.info(f' table exist, {count}')

        return {'count': count}
    
def etl_review_dimension(**kwargs):
    db_host = os.getenv("DATABASE_ENDPOINT")
    db_user = os.getenv("DATABASE_USERNAME")
    db_pw = os.getenv("DATABASE_PASSWORD")
    db_name = os.getenv("DATABASE_NAME")

    dwh_host = os.getenv("DATAWH_ENDPOINT")
    dwh_user = os.getenv("DATAWH_USERNAME")
    dwh_pw = os.getenv("DATAWH_PASSWORD")
    dwh_name = os.getenv("DATAWH_NAME")
    dwh_port = os.getenv("DATAWH_PORT")


    db_datawarehouse = mysql.connector.connect(
        host=dwh_host,
        user=dwh_user,
        passwd=dwh_pw,
        database=dwh_name,
        auth_plugin=dwh_pw
    )

    db_tripadvisor = mysql.connector.connect(
        host=db_host,
        user=db_user,
        passwd=db_pw,
        database=db_name
    )

    engine = create_engine(f'mysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False, future=True)
    db_datawarehouse = engine.connect()

    count = kwargs['ti'].xcom_pull(task_ids='check_dwh_count', key = 'count')
    count = int(count)
    logging.info(type(count))

    # add to review dimension 
    review_sql = f"""
    SELECT ReviewTitle, ReviewText, AuthorContribution, Rating, OverallID
    FROM tripadvisor_reviews
    WHERE OverallID > {count}
    """
    logging.info(review_sql)

    review_df = pd.read_sql(sql=review_sql, con=db_tripadvisor)
    logging.info("passed ")
    review_df['ReviewID'] = review_df['ReviewTitle'].apply(lambda x: str(uuid4())[:12])
    cols = review_df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    review_df = review_df[cols]

    review_df['Rating'] = review_df['Rating'].astype(int)
    review_df['CleanReviewTitle'] = review_df['ReviewTitle'].apply(process_text)
    review_df['CleanReviewText'] = review_df['ReviewText'].apply(process_text)
    review_df['WeightedTitleScore'] = review_df.apply(weighed_title_score, axis = 1)
    review_df['WeightedTextScore'] = review_df.apply(weighed_text_score, axis = 1)
    review_df.drop(['ReviewTitle', 'ReviewText'], axis=1, inplace=True)
    logging.info(review_df)

    review_df.to_sql(name='review', con = db_datawarehouse, if_exists='append')
    logging.info("last")
    db_datawarehouse.commit()
    db_datawarehouse.close()
    db_tripadvisor.close()

def etl_time_dimension(**kwargs):
    db_host = os.getenv("DATABASE_ENDPOINT")
    db_user = os.getenv("DATABASE_USERNAME")
    db_pw = os.getenv("DATABASE_PASSWORD")
    db_name = os.getenv("DATABASE_NAME")

    dwh_host = os.getenv("DATAWH_ENDPOINT")
    dwh_user = os.getenv("DATAWH_USERNAME")
    dwh_pw = os.getenv("DATAWH_PASSWORD")
    dwh_name = os.getenv("DATAWH_NAME")
    dwh_port = os.getenv("DATAWH_PORT")


    db_datawarehouse = mysql.connector.connect(
        host=dwh_host,
        user=dwh_user,
        passwd=dwh_pw,
        database=dwh_name,
        auth_plugin=dwh_pw
    )

    db_tripadvisor = mysql.connector.connect(
	host=db_host,
	user=db_user,
	passwd=db_pw,
	database=db_name
    )

    engine = create_engine(f'mysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False, future=True)
    db_datawarehouse = engine.connect()
    count = kwargs['ti'].xcom_pull(task_ids='check_dwh_count', key = 'count')
    count = int(count)
    logging.info(type(count))

    # Load Time table in sql 
    time_sql = f'''
    SELECT  OverallID,
            tripadvisor_reviews.DateOfStay AS StayDate,
            YEAR(DateOfStay) AS StayDateYear, 
            MONTH(DateOfStay) AS StayDateMonth, 
            Day(DateOfStay) AS StayDateDay, 
            IF((DayOfWeek(DateOfStay) - 1) = 0, 7, DayOfWeek(DateOfStay) - 1) As StayDateDayOfWeek, 
            WEEK(DateOfStay) AS StayDateWeek
    FROM tripadvisor_reviews
    WHERE OverallID > {count}
    '''
    logging.info(time_sql)
    time_df = pd.read_sql(sql=time_sql, con=db_tripadvisor)
    time_df['TimeID'] = time_df['StayDate'].apply(lambda x: str(uuid4())[:12])
    cols = time_df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    time_df = time_df[cols]

    # change staydate to datetime type
    time_df["StayDate"] = pd.to_datetime(time_df['StayDate'], format='%Y-%m-%d')

    # Load Time Dimension 
    time_df.to_sql(name='time', con = db_datawarehouse, if_exists='append')
    db_datawarehouse.commit()
    db_datawarehouse.close()

def etl_fact(**kwargs):
    dwh_host = os.getenv("DATAWH_ENDPOINT")
    dwh_user = os.getenv("DATAWH_USERNAME")
    dwh_pw = os.getenv("DATAWH_PASSWORD")
    dwh_name = os.getenv("DATAWH_NAME")
    dwh_port = os.getenv("DATAWH_PORT")

    db_datawarehouse = mysql.connector.connect(
        host=dwh_host,
        user=dwh_user,
        passwd=dwh_pw,
        database=dwh_name,
        auth_plugin=dwh_pw,

    )
    engine = create_engine(f'mysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False, future=True)
    db_datawarehouse = engine.connect()
    count = kwargs['ti'].xcom_pull(task_ids='check_dwh_count', key = 'count')
    count = int(count)

    # Load Fact Table in dwh
    fact_sql = f'''
    SELECT review.OverallID, review.ReviewID, time.TimeID, review.WeightedTextScore, review.WeightedTitleScore
    FROM review
    INNER JOIN time ON review.OverallID = time.OverallID
    WHERE review.OverallID > {count}
    '''
    fact_df = pd.read_sql(sql = fact_sql, con=db_datawarehouse)
    fact_df["Text_Sentiment"] = fact_df['WeightedTextScore'].apply(threshold)
    fact_df["Title_Sentiment"] = fact_df['WeightedTitleScore'].apply(threshold)
    fact_df.drop(['WeightedTextScore', 'WeightedTitleScore'], axis=1, inplace=True)
    fact_df.to_sql(name='fact', con=db_datawarehouse, if_exists='append')
    db_datawarehouse.commit()
    db_datawarehouse.close()

def set_pk_fk(**kwargs):
    dwh_host = os.getenv("DATAWH_ENDPOINT")
    dwh_user = os.getenv("DATAWH_USERNAME")
    dwh_pw = os.getenv("DATAWH_PASSWORD")
    dwh_name = os.getenv("DATAWH_NAME")
    dwh_port = os.getenv("DATAWH_PORT")

    db_datawarehouse = mysql.connector.connect(
        host=dwh_host,
        user=dwh_user,
        passwd=dwh_pw,
        database=dwh_name,
        auth_plugin=dwh_pw
    )

    engine = create_engine(f'mysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False)
    db_datawarehouse = engine.raw_connection()
    cursor = db_datawarehouse.cursor()

    cursor.execute('ALTER TABLE review ADD PRIMARY KEY (OverallID);')
    cursor.execute('ALTER TABLE time ADD PRIMARY KEY (OverallID);')
    cursor.execute('ALTER TABLE fact ADD PRIMARY KEY (OverallID);')
    cursor.execute('ALTER TABLE fact ADD FOREIGN KEY (OverallID) REFERENCES review(OverallID);')
    cursor.execute('ALTER TABLE fact ADD FOREIGN KEY (OverallID) REFERENCES time(OverallID);')

    db_datawarehouse.commit()
    db_datawarehouse.close()

with DAG(
    'ETL_DAG',
    default_args={
        'depends_on_past': False,
        'email': ['e0726179@u.nus.edu, joellengjd@gmail.com', 'joannelimxiangling@gmail.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='ETL into data warehouse',

    # This particular DAG will run monthly on the 2nd day of the month at 10am.
    schedule_interval='* 10 2 * *',

    start_date=datetime(2024, 2, 28),
    dagrun_timeout=timedelta(minutes=30),
    catchup=False,

) as dag:
    # define tasks by instantiating operators
    set_overall_id = PythonOperator(
        task_id='set_overall_id',
        python_callable= set_overall_id,
        dag=dag
    )

    check_dwh_count = PythonOperator(
        task_id = 'check_dwh_count',
        python_callable = check_dwh_count,
        provide_context=True,
        dag=dag

    )

    etl_review_dimension = PythonOperator(
        task_id='etl_review_dimension',
        python_callable= etl_review_dimension,
        dag=dag
    )

    etl_time_dimension = PythonOperator(
        task_id='etl_time_dimension',
        python_callable= etl_time_dimension,
        dag=dag
    )

    etl_fact = PythonOperator(
        task_id='etl_fact',
        python_callable= etl_fact,
        dag=dag
    )

    set_pk_fk = PythonOperator(
        task_id='set_pk_fk',
        python_callable= set_pk_fk, 
        dag=dag
    )

    set_overall_id >> check_dwh_count >> etl_review_dimension >> etl_time_dimension >> etl_fact >> set_pk_fk
