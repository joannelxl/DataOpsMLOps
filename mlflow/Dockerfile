FROM python:3.11
WORKDIR /app
COPY . /app

RUN pip install -r pre-requisite.txt

EXPOSE 5000

CMD ["python", "text_blob.py"]