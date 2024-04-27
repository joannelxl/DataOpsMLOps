# Import for database connection and other basic libraries
import mysql.connector
import pandas as pd
import numpy as np
import os
import re
from sqlalchemy import create_engine


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')

from imblearn.over_sampling import RandomOverSampler

# Import libraries for LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchinfo import summary
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")


import yaml
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
engine = create_engine(f'mysql://{dwh_user}:{dwh_pw}@{dwh_host}:{dwh_port}/{dwh_name}', echo=False, future=True)

db_datawarehouse = engine.connect()

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Reading the data
df = pd.read_sql_table('review', db_datawarehouse).copy()
df2 = pd.read_sql_table('fact', db_datawarehouse).copy()
df3 = pd.read_sql_table('time', db_datawarehouse).copy()
df, df2, df3 = df.drop(columns=['index']), df2.drop(columns=['index']), df3.drop(columns=['index'])
df4 = pd.merge(df, df2, on='ReviewID', how='inner')
final_df = pd.merge(df4, df3, on='TimeID', how='inner')
print(final_df.columns.tolist())
# Split into train, test, val set
data = final_df[['CleanReviewText', 'StayDateYear', 'Text_Sentiment']]
train_data = data[data['StayDateYear'].apply(lambda x: int(x)) < 2022]
test_data = data[data['StayDateYear'].apply(lambda x: int(x)) >= 2022]
X_trainset, y_trainset = train_data['CleanReviewText'].values, train_data['Text_Sentiment'].values
X_test_val, y_test_val = test_data['CleanReviewText'].values, test_data['Text_Sentiment'].values
X_testset, X_valset, y_testset, y_valset = train_test_split(X_test_val, y_test_val, test_size=0.5)

print(f'Shape of train data is {X_trainset.shape} and {y_trainset.shape}')
print(f'Shape of test data is {X_testset.shape} and {y_testset.shape}')
print(f'Shape of val data is {X_valset.shape} and {y_valset.shape}')

# Feature engineering for LSTM
def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def process_text(text):
    
    # Initialise
    lemmatizer = WordNetLemmatizer()
    edited_stop_words = stopwords.words('english')
    processed_text = " "
    
    # Process input
    text_lower = text.lower()
    word = word_tokenize(text_lower)
    
    # Alphabetical Tokens
    alphabetic_tokens = [preprocess_string(word) for word in word if re.match('^[a-zA-Z]+$', word)]
    
    print(alphabetic_tokens)

    # Edit stopwords list
    edited_stop_words.remove('no')
    edited_stop_words.remove('not')
    edited_stop_words.remove("wouldn't")
    edited_stop_words.remove('wouldn')
    edited_stop_words.remove("couldn't")
    edited_stop_words.remove('couldn')
    edited_stop_words.remove("against")
    
    # Remove stopwords from text and lemmatize
    stop_words = set(edited_stop_words)
    
    lem_words = []
    for word in alphabetic_tokens:
        if word not in stop_words:
            lem_words.append(lemmatizer.lemmatize(word))
    
    # Join the list of words
    processed_text = processed_text.join(lem_words)     #print(edited_stop_words)

    return processed_text

def tokenize(x_train, x_test, x_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        # print(type(sent)) 'str'
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of 1000 most common words.
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tokenize
    final_list_train, final_list_test, final_list_val = [], [], []
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_test:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    for sent in x_val:
            final_list_val.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                    if preprocess_string(word) in onehot_dict.keys()])

    return np.array(final_list_train, dtype=object), np.array(final_list_test, dtype=object), np.array(final_list_val, dtype=object), onehot_dict

X_train, X_test, X_val , vocab = tokenize(X_trainset, X_testset, X_valset)
y_train, y_test, y_val = y_trainset, y_testset, y_valset

# Padding for the input sequences
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


X_train_pad = padding_(X_train, 500)
X_test_pad = padding_(X_test, 500)
X_val_pad = padding_(X_val, 500)

# Random oversampling to address class imbalance
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train_pad, y_train)
print(f'Shape of train data is {X_train_res.shape} and {y_train_res.shape}')

# TensorDataset
train_data = TensorDataset(torch.from_numpy(X_train_res), torch.from_numpy(y_train_res))
test_data = TensorDataset(torch.from_numpy(X_test_pad), torch.from_numpy(y_test))
val_data = TensorDataset(torch.from_numpy(X_val_pad), torch.from_numpy(y_val))

# Batch size
batch_size = 64

# DataLoader
train_loader = DataLoader(train_data, shuffle=True, batch_size=64, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=64, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=64, drop_last=True)

# LSTM class
class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim):
        super(SentimentRNN,self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out

        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 64]; 64 is the embedding_dim defined below.
        
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Calling lstm_out.contiguous()to ensure the output tensor from the LSTM is contiguous before performing the view operation.
        # reshapes the lstm_out tensor to have 2D layer with a shape of (batch_size * sequence_length, hidden_dim).
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels, this is very important for an output of a sentiment score!!!

        # return last sigmoid output and hidden state
        return sig_out, hidden



    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256
model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim).to(device)

lr = 0.001
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
metric_fn = Accuracy(task="binary").to(device)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

# set the gradient clipping threshold and number of training epochs
clip = 5
epochs = 5

def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch, device):
    model.train()
    # initialize hidden state
    h = model.init_hidden(batch_size)
    for batch, (inputs, labels) in enumerate(dataloader):

        inputs, labels = inputs.to(device), labels.to(device)
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # Clear the gradients
        model.zero_grad()
        
        # Perform a forward pass through the model
        output, h = model(inputs, h)

        # calculate the loss and perform backpropogation
        loss = loss_fn(output.squeeze(), labels.float())
        loss.backward()

        # calculating accuracy
        accuracy = metrics_fn(output, labels)
        # Gradient Clipping: `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Optimizer Step: Update the model's parameters using the optimizer
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
            print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, device):
    model.eval()
    # Initialize Hidden States
    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0

    # Loop Through Test Data
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)
            # Forward Pass
            output, val_h = model(inputs, val_h)

            # Calculate Loss and Metrics(Accuracy)
            val_loss = loss_fn(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
            accuracy = metrics_fn(output,labels)
            val_acc += accuracy
    
    eval_loss = np.mean(val_losses)
    eval_accuracy = val_acc/len(dataloader)
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy*100:2f}", step=epoch)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy*100:.2f}, Avg loss: {eval_loss:2f} \n")

mlflow.set_tracking_uri(uri="http://localhost:9080")
mlflow.set_experiment("MLFlow for Project - LSTM")


# Sample input from train_loader
sample_inputs, _ = next(iter(train_loader))

# Init hidden
h = model.init_hidden(batch_size)

# Get sample output
sample_outputs, h = model(sample_inputs, h)

# To numpy
sample_inputs_np = sample_inputs.numpy()
sample_outputs_np = sample_outputs.detach().numpy()

# Infer the signature 
signature = infer_signature(sample_inputs_np, sample_outputs_np)

with mlflow.start_run() as run:
    mlflow.set_tag("Model", "LSTM")
    params = {
        "epochs": epochs,
        "clip": clip,
        "learning_rate": lr,
        "batch_size": batch_size,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": optimizer,
        "device": device
    }
    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("LSTM_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("LSTM_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, metric_fn, optimizer, epoch=t, device=device)
        evaluate(test_loader, model, loss_fn, metric_fn, epoch=t, device=device)
    
    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "LSTM", signature=signature)

# Inference after loading the logged model
model_uri = f"runs:/{run.info.run_id}/LSTM"
loaded_model = mlflow.pytorch.load_model(model_uri)