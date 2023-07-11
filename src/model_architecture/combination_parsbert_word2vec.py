import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import gc
from tqdm import tqdm
from pathlib import Path
import random
import pandas as pd
import json
import numpy as np
from transformers import BertModel, BertTokenizerFast
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[2]
models_dir = root_dir / "models"

data_count = 400
dimension = 768
experiment_model_architecture_dir = root_dir / "experiments" / "model_architecture"
experiment_model_architecture_dir.mkdir(exist_ok=True, parents=True)


def make_trait(row):
    row["trait_0"] = 0.0 if row["mbti_result"][0] == "I" else 1.0
    row["trait_1"] = 0.0 if row["mbti_result"][1] == "N" else 1.0
    row["trait_2"] = 0.0 if row["mbti_result"][2] == "T" else 1.0
    row["trait_3"] = 0.0 if row["mbti_result"][3] == "J" else 1.0
    return row


print("loading dataset ...")
root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
log_file = root_dir / "run.log"
df = pd.read_json(dataset_file.as_posix())
df = df.iloc[:data_count]
print("preparing dataset ...")
df = df.apply(make_trait, axis=1)
input_data = df
labels = df["trait_0"]

train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size=0.2,
                                                                    random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25,
                                                                  random_state=42)

loss_stat = {}
acc_stat = {}


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 1000)
        self.dropout_1 = nn.Dropout(0.3)
        self.fc_2 = nn.Linear(1000, 200)
        self.dropout_2 = nn.Dropout(0.3)
        self.fc_3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.dropout_1(self.fc_1(x)))
        x = F.relu(self.dropout_2(self.fc_2(x)))
        x = self.fc_3(x)
        return F.sigmoid(x)


# Set the random seed for reproducibility
torch.manual_seed(42)


def train(X_data, y_data, model_name, input_dim=768):
    # Split the data into train, test, and validation sets
    X_train, y_train = X_data[0], y_data[0]
    X_test, y_test = X_data[1], y_data[1]
    X_val, y_val = X_data[2], y_data[2]
    # Create the model
    model = BinaryClassifier(input_dim).cuda()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 60
    batch_size = 32
    epochs_loss = {"train": [], "val": [], "test": []}
    epochs_acc = {"train": [], "val": [], "test": []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        indices = list(range(len(X_train)))
        random.shuffle(indices)
        X_train_shuffled = torch.stack([X_train[i] for i in indices])
        y_train_shuffled = torch.stack([y_train[i] for i in indices])
        for i in range(0, len(X_train_shuffled), batch_size):
            inputs = X_train_shuffled[i:i + batch_size]
            labels = y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()

            outputs = model(inputs).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss}")

        with torch.no_grad():
            train_outputs = model(X_train).float()
            train_loss = criterion(train_outputs, y_train)
            train_predicted_labels = train_outputs.round()
            train_accuracy = (y_train == train_predicted_labels).float().mean()
            epochs_loss["train"].append(train_loss.to("cpu").item())
            epochs_acc["train"].append(train_accuracy.to("cpu").item())

            val_outputs = model(X_val).float()
            val_loss = criterion(val_outputs, y_val)
            val_predicted_labels = val_outputs.round()
            val_accuracy = (y_val == val_predicted_labels).float().mean()
            epochs_loss["val"].append(val_loss.to("cpu").item())
            epochs_acc["val"].append(val_accuracy.to("cpu").item())

            test_outputs = model(X_test).float()
            test_loss = criterion(test_outputs, y_test)
            test_predicted_labels = test_outputs.round()
            test_accuracy = (y_test == test_predicted_labels).float().mean()
            epochs_loss["test"].append(test_loss.to("cpu").item())
            epochs_acc["test"].append(test_accuracy.to("cpu").item())

    torch.save(model.state_dict(),
               f"{models_dir.as_posix()}/model_architecture_{model_name}.pth")

    print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return epochs_loss, epochs_acc


def extract_vector(vectors, tokens_w2v, query_word, log_file):
    word_exists = tokens_w2v.get(query_word, -1)
    if word_exists == -1:
        # print(f"the qurey word {query_word} doesn't exist")
        with open(log_file.as_posix(), "a+", encoding="utf-8") as f:
            f.write("---------------\n")
            f.write(
                f"the query word {query_word} to get vector from doesn't exist\n")
            f.write("the error occured in src/word2vec/query.py file\n")
        return False
    else:
        word_vector = vectors[tokens_w2v[query_word]]
        # print(f"word vector of {query_word} is : {vectors[tokens_w2v[query_word]]}")
        return word_vector


def extract_w2v(tweets, vectors, tokens_word2vec, log_file):
    curr_vector = np.zeros(10)
    top_k = 50
    for tweet in tweets[:top_k]:
        for word in tweet.split():
            word_vector = extract_vector(
                vectors, tokens_word2vec, word, log_file)
            if word_vector is not False:
                curr_vector += word_vector
    curr_vector = np.pad(curr_vector, (0, dimension - 10),
                         'constant', constant_values=0)
    return curr_vector


def load_word2vec_model():
    root_dir = Path(__file__).resolve().parents[2]
    model_file = root_dir / "models" / "all.word2vec.npy"
    token_file = root_dir / "experiments" / "word2vec" / "all_tokens.json"
    logs_dir = root_dir / "logs"
    log_file = logs_dir / "word2vec.log"

    vectors = np.load(model_file.as_posix())
    with open(token_file.as_posix(), "r") as f:
        tokens_word2vec = json.load(f)

    return vectors, tokens_word2vec, log_file


def tweets_embedding(tweets, tokenizer, embeddings_dict, dimension):
    top_k = 128
    tokenized_input = tokenizer.batch_encode_plus(tweets[:top_k], add_special_tokens=True, truncation=True,
                                                  padding=True)
    input_ids_batch = torch.tensor(tokenized_input["input_ids"])
    embeddings = embeddings_dict[input_ids_batch]
    tweet_sum = torch.sum(embeddings, axis=1)
    total_sum = torch.sum(tweet_sum, axis=0)

    return total_sum


def pars_bert_feature(df: pd.DataFrame):
    tokenizer = BertTokenizerFast.from_pretrained(
        "HooshvareLab/bert-fa-base-uncased")
    bert = BertModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    embedding_matrix = bert.embeddings.word_embeddings.weight
    feature_dimension = embedding_matrix.shape[1]
    # extract this tensors from pytorch and convert it to numpy
    return torch.stack(
        df["tweets"].apply(lambda x: tweets_embedding(x, tokenizer, embedding_matrix, feature_dimension)).tolist())


def word2vec_feature(df: pd.DataFrame):
    vectors, tokens_word2vec, log_file = load_word2vec_model()
    return df["tweets"].apply(lambda x: extract_w2v(x, vectors, tokens_word2vec, log_file))


def word2vec_ParsBERT_combined():
    torch.cuda.empty_cache()
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    print("extract word2vec feature ...")
    for data in X_datas:
        word_length_vector_1 = np.stack(word2vec_feature(data).to_numpy())
        word_length_vector_1 = torch.from_numpy(word_length_vector_1).float()
        word_length_vector_2 = pars_bert_feature(data).detach().float()
        word_length_vector_2 -= word_length_vector_2.min(1, keepdim=True)[0]
        word_length_vector_2 /= word_length_vector_2.max(1, keepdim=True)[0]
        word_length_vector = (word_length_vector_1 + word_length_vector_2) / 2
        datas_X.append(word_length_vector)
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]
    print("train combined Parsbert and word2vec embedding model ...")

    loss, acc = train(X_datas, y_datas, "combined_parsbert_word2vec")
    loss_stat["combined_parsbert_word2vec"] = loss
    acc_stat["combined_parsbert_word2vec"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(
            f"model architecture engineering with combined_parsbert_word2vec is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


def main():
    word2vec_ParsBERT_combined()
    result = {"acc": acc_stat, "loss": loss_stat}
    with open(f"{experiment_model_architecture_dir.as_posix()}/results.json", "w") as f:
        json.dump(result, f)
    
    

    stats_dir = (root_dir / "stats").as_posix()

    with open(f"{experiment_model_architecture_dir.as_posix()}/results.json", "r") as f:
        results = json.load(f)

    acc = results["acc"]
    loss = results["loss"]

    for key_0, value_0 in acc.items():
        for key_1, value_1 in value_0.items():
            plt.plot(value_1, label=f"{key_0}-{key_1}")
            plt.title("accuracy based on model architecture ParsBERT and word2vec")
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.legend()
    plt.savefig(f"{stats_dir}/acc_model_architecture.png")
    plt.figure()
    for key_0, value_0 in loss.items():
        for key_1, value_1 in value_0.items():
            plt.plot(value_1, label=f"{key_0}-{key_1}")
            plt.title("loss based on feature")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.legend()
    plt.savefig(f"{stats_dir}/loss_model_architecture.png")

main()
