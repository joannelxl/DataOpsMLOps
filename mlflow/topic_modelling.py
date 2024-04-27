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
import yaml
import nltk
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag
import mlflow
from mlflow.models import infer_signature
import mlflow.pyfunc

#LOAD DATA
with open('data/db_info.yml', 'r') as file:
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

def preprocess_text(text):
    is_noun = lambda pos: pos[:2] == 'NN'
    text = text.split()
    nouns = [word for (word, pos) in pos_tag(text) if is_noun(pos)]
    return nouns

df['ProcessedText'] = df['CleanReviewText'].apply(preprocess_text)
texts = df["ProcessedText"].to_numpy()
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

class LDA(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass

    def predict(self, model_input, params=None):
        lda_model = LdaModel(corpus=corpus,
                    id2word=dictionary,
                    num_topics=5,
                    random_state=42,
                    update_every=1,
                    chunksize=100,
                    passes=5,
                    alpha='auto',
                    per_word_topics=True)
        topics = ""
        for idx, topic in lda_model.print_topics(-1):
            topics += f"Topic: {idx} \nWords: {topic}\n"
        return topics

model_path = "lda"
# Define the signature associated with the model



mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("MLFlow for Project - Topic Modelling")

with mlflow.start_run() as run:
    mlflow.set_tag("Model", "Topic Modelling")

    artifact_uri = run.info.artifact_uri

    model = LDA()
    y_pred = model.predict(texts)
    signature = infer_signature(texts, model.predict(texts))
    
    mlflow.log_text(y_pred, "file.txt")
    file_content = mlflow.artifacts.load_text(artifact_uri + "/file.txt")
    print(file_content)

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=model,
        input_example=texts,
        signature=signature,
        pip_requirements=None,
    )


loaded_model = mlflow.pyfunc.load_model(model_path)