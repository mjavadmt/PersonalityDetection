import pandas as pd
from pathlib import Path
from utils import *
from model import *
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from datetime import datetime
import json

data_count = 400
dimension = 768
experiment_feat_engineering_dir = root_dir / "experiments" / "feat_engineering"
experiment_feat_engineering_dir.mkdir(exist_ok=True, parents=True)


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


# feature number 1 : sentence_length
def feature_1():
    global loss_stat, acc_stat
    print("train sentence length model ...")
    torch.cuda.empty_cache()
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    applied_datas = [torch.tensor(
        extract_sentence_length(i).values).float() for i in X_datas]
    unsqueezed_data = [i.unsqueeze(1) for i in applied_datas]
    padded_data = [F.pad(i, pad=(0, dimension - 1, 0, 0))
                   for i in unsqueezed_data]
    X_datas = [i.cuda() for i in padded_data]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]

    loss, acc = train(X_datas, y_datas, "sentence_length")
    loss_stat["sentence-length"] = loss
    acc_stat["sentence-length"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with sentence_length is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


# feature number 2 : word_length
def feature_2():
    print("train word length model ...")
    torch.cuda.empty_cache()
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    for data in X_datas:
        word_length_vector = np.stack(extract_word_length(data).to_numpy())
        word_length_vector = torch.from_numpy(word_length_vector).float()
        datas_X.append(word_length_vector)
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]

    loss, acc = train(X_datas, y_datas, "word_length")
    loss_stat["word-length"] = loss
    acc_stat["word-length"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with word_length is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


# feature number 3 : words
def feature_3():
    torch.cuda.empty_cache()
    print("assigning tokens for unigram words ...")
    assign_tokens(df)
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    print("train word model ...")
    for data in X_datas:
        word_length_vector = np.stack(words_uni_gram(data).to_numpy())
        word_length_vector = torch.from_numpy(word_length_vector).float()
        # normalizing
        word_length_vector -= word_length_vector.min(1, keepdim=True)[0]
        word_length_vector /= word_length_vector.max(1, keepdim=True)[0]
        datas_X.append(word_length_vector)
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]

    loss, acc = train(X_datas, y_datas, "word_unigram")
    loss_stat["word-unigram"] = loss
    acc_stat["word-unigram"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with word_unigram is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


# feature number 4 : words bigram
def feature_4():
    torch.cuda.empty_cache()
    # assign_tokens(df)
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    print("train bi-gram word model ...")
    for data in X_datas:
        word_length_vector = np.stack(words_bi_gram(data).to_numpy())
        word_length_vector = torch.from_numpy(word_length_vector).float()
        word_length_vector -= word_length_vector.min(1, keepdim=True)[0]
        word_length_vector /= word_length_vector.max(1, keepdim=True)[0]
        datas_X.append(word_length_vector)
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]

    loss, acc = train(X_datas, y_datas, "word_bigram")
    loss_stat["word-bigram"] = loss
    acc_stat["word-bigram"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with word_bigram is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


# feature number 5 : word2vec
def feature_5():
    torch.cuda.empty_cache()
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    print("extract word2vec feature ...")
    for data in X_datas:
        word_length_vector = np.stack(word2vec_feature(data).to_numpy())
        word_length_vector = torch.from_numpy(word_length_vector).float()
        datas_X.append(word_length_vector)
    print("train word2vec model ...")
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]

    loss, acc = train(X_datas, y_datas, "word2vec")
    loss_stat["word2vec"] = loss
    acc_stat["word2vec"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with word2vec is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


# feature number 6 : word2vec bigram
def feature_6():
    torch.cuda.empty_cache()
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    print("extract word2vec feature ...")
    for data in X_datas:
        word_length_vector = np.stack(word2vec_bi_gram(data).to_numpy())
        word_length_vector = torch.from_numpy(word_length_vector).float()
        datas_X.append(word_length_vector)
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]
    print("train word2vec bigram model ...")

    loss, acc = train(X_datas, y_datas, "word2vec-bigram")
    loss_stat["word2vec-bigram"] = loss
    acc_stat["word2vec-bigram"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with word2vec-bigram is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


# feature number 7 : ParsBERT embeddings
def feature_7():
    torch.cuda.empty_cache()
    X_datas = [train_data, test_data, val_data]
    y_datas = [train_labels, test_labels, val_labels]
    datas_X = []
    print("extract ParsBERT feature ...")
    for data in X_datas:
        word_length_vector = pars_bert_feature(data).detach().float()
        word_length_vector -= word_length_vector.min(1, keepdim=True)[0]
        word_length_vector /= word_length_vector.max(1, keepdim=True)[0]
        datas_X.append(word_length_vector)
    X_datas = [i.cuda() for i in datas_X]
    y_datas = [torch.tensor(i.values).unsqueeze(
        1).float().cuda() for i in y_datas]
    print("train ParsBERT embedding model ...")

    loss, acc = train(X_datas, y_datas, "ParsBERT_embeddings")
    loss_stat["ParsBERT-embeddings"] = loss
    acc_stat["ParsBERT-embeddings"] = acc

    with open(log_file.as_posix(), "a+") as f:
        f.write("-----------------------------\n")
        curr_datetime = str(datetime.now())
        f.write(f"at time : {curr_datetime[:-7]}\n")
        f.write(f"feature engineering with ParsBERT-embeddings is trained\n")

    torch.cuda.empty_cache()
    print("-" * 50)


features = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
print("running all models ...")
for feat in features:
    feat()

result = {"acc": acc_stat, "loss": loss_stat}
with open(f"{experiment_feat_engineering_dir.as_posix()}/results.json", "w") as f:
    json.dump(result, f)
