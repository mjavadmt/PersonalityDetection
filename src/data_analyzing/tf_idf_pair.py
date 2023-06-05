import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def make_trait(row):
    row["trait_0"] = row["mbti_result"][0]
    row["trait_1"] = row["mbti_result"][1]
    row["trait_2"] = row["mbti_result"][2]
    row["trait_3"] = row["mbti_result"][3]
    return row


root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()

dataframe = pd.read_json(dataset_file)
dataframe = dataframe.apply(make_trait, axis=1)
freq_on_labels = {}
for trait in range(4):
    curr_label_str = f"trait_{trait}"
    curr_df = dataframe[["tweets", curr_label_str]]
    grouped_curr_df = curr_df.groupby(curr_label_str)
    documents = []
    frequencies = {}
    labels = []
    for label, frame in grouped_curr_df:
        labels.append(label)
        document_plain_text = " ".join(list(frame["tweets"].apply(lambda x: " ".join(x))))
        documents.append(document_plain_text)

    vectorizer = TfidfVectorizer()

    # Compute the TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Print the TF-IDF scores
    feature_names = np.array(vectorizer.get_feature_names())

    num_popular_word = 20
    popular_words_indices = np.argsort(tfidf_matrix.toarray(), axis=1)[:, -num_popular_word:]

    for i, label in enumerate(labels):
        print(f"{label} popular words are : ")
        print(list(feature_names[popular_words_indices[i]]))
        freq_on_labels[label] = list(feature_names[popular_words_indices[i]])

df = pd.DataFrame.from_dict(freq_on_labels, orient="index")
df.to_csv(f"{stats_dir}/tf-idf_pair.csv", encoding="utf-8")
