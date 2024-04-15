import mysql.connector
import pandas as pd
import os
from sqlalchemy import create_engine
import re
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel, LsiModel, HdpModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import yaml
import nltk
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import mlflow
from mlflow.models import infer_signature
import mlflow.pyfunc

#LOAD DATA
with open('../data/db_info.yml', 'r') as file:
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

db_datawarehouse = mysql.connector.connect(
	host=dwh_host,
	user=dwh_user,
	passwd=dwh_pw,
	database=dwh_name,
    auth_plugin=dwh_pw
)
engine = create_engine(f'mysql+pymysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False)

dwh = engine.connect()

review_sql = '''
SELECT * FROM review ORDER BY OverallID
'''
df = pd.read_sql(sql=review_sql, con=db_datawarehouse)
dwh.close()

reviews = df["CleanReviewText"].to_numpy()


class Vader(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass

    def predict(self, model_input, params=None):
        res = []
        analyzer = SentimentIntensityAnalyzer()
        for each in model_input:
            res.append(analyzer.polarity_scores(each)["compound"])
        return res

model_path = "vader"
model = Vader()
y_pred = model.predict(reviews)
y_pred = [0 if pred<=0 else 1 for pred in y_pred]
y_truth = [0 if pred<=0 else 1 for pred in df["TextBlob_Review"]]
accuracy = accuracy_score(y_truth, y_pred)

# Define the signature associated with the model
signature = infer_signature(reviews, model.predict(reviews))


mlflow.set_tracking_uri(uri="http://localhost:9080")
mlflow.set_experiment("MLFlow for Project - Vader")

with mlflow.start_run():
    mlflow.set_tag("Model", "Vader")

    mlflow.log_metric("accuracy", accuracy)

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=model,
        input_example=reviews,
        signature=signature,
        pip_requirements=None,
    )


loaded_model = mlflow.pyfunc.load_model(model_path)