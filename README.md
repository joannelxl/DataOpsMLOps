# Sentiment Analysis and Topic Modelling on Marina Bay Sands (MBS) Hotel Reviews

## About the Project
The goal of this project is to generate customer insights on their satisfactions/dissatisactions of staying in MBS hotel. This is done using scraped data from tripadvisor. To achieve the objectives of the app, we used DataOps and MLOps concepts and created a frontend dashboard.

## Tech Stack:
- Frontend: Streamlit
- Database: MySQL using AWS Relational Database Services (RDS)
- Data Orchestration: Apache Airflow
- MLOps: MLflow
- Containerisation: Docker

## Project Setup
1. Input relevant files:
- Place .env file in same level as READMe.md
- Place db_info.yml file in `streamlit_app` folder
2. Build the image from the dockerfile (Only needs to be run once to build):
`docker build -t streamlit . ` 
3. Start container to connect to localhost:8501 (Run this command whenever to open the app):
`docker run -p 8501:8501 streamlit` 

## Star Schema 
![image](https://github.com/e0727131/BT4301_GP09/assets/79855907/b60fb877-8f26-48d4-88fd-868c67e7c74e)

## Contributors
Gao Heng (@foxiegh) \
Hom Lim Jun How (@e0727131)\
Leng Jin De Joel (@joelleng)\
Lim Xiang Ling (@joannelxl)\
Myo Nyi Nyi (@myonster)


