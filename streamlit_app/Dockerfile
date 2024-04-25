FROM python:3.11

WORKDIR /streamlit_app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader vader_lexicon
RUN python3 -m nltk.downloader averaged_perceptron_tagger

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]