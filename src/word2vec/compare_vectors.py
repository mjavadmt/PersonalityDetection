import json
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from query import extract_vector
import pandas as pd

traits = [["I", "E"], ["S", "N"], ["T", "F"], ["J", "P"]]
trait = 0

root_dir = Path(__file__).resolve().parents[2]
model_file_0 = root_dir / "models" / \
    f"{traits[trait][0]}_{trait}.word2vec.npy"
model_file_1 = root_dir / "models" / \
    f"{traits[trait][1]}_{trait}.word2vec.npy"
token_file_0 = root_dir / "experiments" / "word2vec" / \
    f"{traits[trait][0]}_{trait}_tokens.json"
token_file_1 = root_dir / "experiments" / "word2vec" / \
    f"{traits[trait][1]}_{trait}_tokens.json"
logs_dir = root_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "word2vec.log"
stats_dir = (root_dir / "stats").as_posix()

tokens_0 = {}
tokens_1 = {}

vectors_0 = np.load(model_file_0.as_posix())
vectors_1 = np.load(model_file_1.as_posix())
with open(token_file_0.as_posix(), "r") as f:
    tokens_0 = json.load(f)

with open(token_file_1.as_posix(), "r") as f:
    tokens_1 = json.load(f)


def compare_labels_vectors(query_word):
    print("In I trait word vectors is : ")
    vector_0 = extract_vector(vectors_0, tokens_0, query_word, log_file)
    print()
    print("In E trait word vector is")
    vector_1 = extract_vector(vectors_1, tokens_1, query_word, log_file)
    print()
    if vector_0 is False or vector_1 is False:
        print("vectors doesnt exist in at least one model")
        return
    else:
        cos_sim = round(np.dot(vector_0, vector_1) /
                        (norm(vector_0) * norm(vector_1)), 5)
        print(f"cosine similarity between vectors of two model is {cos_sim}")
        return cos_sim


def bias_computer(vectors, tokens):
    w1 = "مرد"
    w2 = "زن"
    w3 = "ترس"

    vector_w1 = extract_vector(vectors, tokens, w1, log_file)
    vector_w2 = extract_vector(vectors, tokens, w2, log_file)
    vector_w3 = extract_vector(vectors, tokens, w3, log_file)

    cos_sim_13 = round(np.dot(vector_w1, vector_w3) /
                        (norm(vector_w1) * norm(vector_w3)), 5)
    
    cos_sim_23 = round(np.dot(vector_w2, vector_w3) /
                        (norm(vector_w2) * norm(vector_w3)), 5)
    
    return cos_sim_13, cos_sim_23

def handle_neg_values(value):
    if value < 0:
        value = f"{abs(value)}-"
    else:
        value = f"{value}"
    return value


query_words = ["شما", "خونه", "اتفاق", "هیجان", "تنهایی"]

E_I_comparer = {}
for word in query_words:
    sim = compare_labels_vectors(word)
    
    E_I_comparer[word] = handle_neg_values(sim)




comparer_series = pd.Series(E_I_comparer)
comparer_series.to_csv(f"{stats_dir}/word2vec_compare.csv", encoding="utf-8")

I_bias_0, I_bias_1 = bias_computer(vectors_0, tokens_0)
E_bias_0, E_bias_1 = bias_computer(vectors_1, tokens_1)
df = pd.DataFrame()
df.loc["I", "مرد-ترس"] = handle_neg_values(I_bias_0)
df.loc["I", "زن-ترس"] = handle_neg_values(I_bias_1)
df.loc["E", "مرد-ترس"] = handle_neg_values(E_bias_0)
df.loc["E", "زن-ترس"] = handle_neg_values(E_bias_1)

df.to_csv(f"{stats_dir}/word2vec_bias.csv", encoding="utf-8")
