from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from sqlalchemy import create_engine
import time
import csv
import random
import pandas as pd



with DAG(
    'ingest_data',
    default_args={
        'depends_on_past': False,
        'email': ['gaoheng@u.nus.edu'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='DAG to ingest tripadvisor data month by month',
    schedule_interval=None,

    start_date=datetime(2024, 2, 28),
    dagrun_timeout=timedelta(seconds=5),
    catchup=False,
    tags = ["bt4301"]
) as dag:
    def scrape_reviews(**kwargs):
        USER_AGENTS = ("--user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'",
                "--user-agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'",
                "--user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'",
                "--user-agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'",
                "--user-agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'",
                "--user-agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'",
                "--user=agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'")

        STATE = random.randint(0, 7)

        USER_AGENT = '"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"'
        SEC_CH_UA = '\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"'
        REFERER = "https://www.google.com"

        def interceptor(request):
            del request.header["user-agent"]
            request.header["user-agent"] = USER_AGENT
            request.header["sec-ch-ua"] = SEC_CH_UA
            request.header["referer"] = REFERER

        chrome_options = Options()
        # chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument(f"{USER_AGENTS[0]}")

        # DRIVER
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.request_interceptor = interceptor
        driver.maximize_window()
        # driver.add_argument("--user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'")

        # WebDriverWait to wait for elements to be rendered
        WAIT = WebDriverWait(driver, 60)

        #Get the page url
        page_url = get_page_url()

        driver.get(page_url)

        sleepTime = random.randint(3, 10)
        time.sleep(sleepTime) 

        review_containers = driver.find_elements(By.XPATH, "//div[contains(@class, 'azLzJ') and contains(@class, 'MI') and contains(@class, 'Gi') and @data-test-target='HR_CC_CARD']")
        # review_containers = WAIT.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'azLzJ') and contains(@class, 'MI') and contains(@class, 'Gi') and @data-test-target='HR_CC_CARD']")))

        if not review_containers:
            print("No more reviews found or a loading issue occurred.")
            return False
        column_names = ["review_title", "review_text", "review_date"]
        review_df = pd.DataFrame(column_names = column_names)
        for review in review_containers:
            try:
                epsilon = random.randint(1, 10)
                review_title = review.find_element(By.XPATH, ".//div[contains(@class, 'joSMp')]/a/span/span").text
                review_text = review.find_element(By.XPATH, ".//div[contains(@class, 'yJgrn')]/div/div/span/span").text
                # review_date = review.find_element(By.XPATH, ".//div[contains(@class, 'sQLyb')]/span/span[contains(text(),'Date of stay:')]").text.replace('Date of stay: ', '')
                review_date = review.find_element(By.XPATH, ".//div[contains(@class, 'sQLyb')]/span").text.replace('Date of stay: ', '')
                # review_title = WAIT.until(EC.element_to_be_clickable((By.XPATH, ".//div[contains(@class, 'joSMp')]/a/span/span").text))
                # review_text = WAIT.until(EC.element_to_be_clickable((By.XPATH, ".//div[contains(@class, 'yJgrn')]/div/div/span/span").text))
                # review_date = WAIT.until(EC.element_to_be_clickable((By.XPATH, ".//div[contains(@class, 'sQLyb')]/span/span[contains(text(),'Date of stay:')]".text.replace('Date of stay: ', ''))))
                time.sleep(sleepTime + epsilon)
                review_df.loc[len(review_df.index)] = [review_title, review_text, review_date]
            except Exception as e:
                print(f"Error extracting review: {e}")       
        #Retrieve the date of the last extracted data in database
        curr_period = get_curr_period()
        
        #Filter to only get data with date past the last extracted data in database 
        df = df[df["review_date"] > curr_period]

        engine = get_engine()
        connection_name = engine.connect()

        #Check the current name of the table in the database
        review_df.to_sql(name="reviews", con=connection_name, if_exists='append', index=False)
        connection_name.commit()

        return True
    def get_page_url():
        # Static user agents to avoid web crawling detection on the website
        base_url = "https://www.tripadvisor.com.sg/Hotel_Review-g294265-d1770798-Reviews"
        offset_pattern = "-or{}-Marina_Bay_Sands-Singapore.html"
        page_url = f"{base_url}{offset_pattern.format(offset_pattern)}"
        return page_url
    
    def get_curr_period():
        engine = get_engine()
        str_sql = '''SELECT `Review Date` from reviews ORDER BY `Review Date` DESC LIMIT 1'''
        with engine.connect() as connection:
            result = connection.execute(str_sql)
        return result[0]
    
    def get_engine():
        engine = create_engine("mysql+pymysql://admin:bt4301db1@bt4301-db1.cxmwooi4whw1.ap-southeast-1.rds.amazonaws.com:3306/BT4301_G09_Database")
        return engine

    def last_task(**kwargs):
        print("end")

    scrape_reviews_task = PythonOperator(task_id="scrape_reviews", python_callable=scrape_reviews)

    do_last_task = PythonOperator(
        task_id='do_last_task',
        python_callable=last_task,
        trigger_rule='one_success'
    )

    scrape_reviews_task >> do_last_task

