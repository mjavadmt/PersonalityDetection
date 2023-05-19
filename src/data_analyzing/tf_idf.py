import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()

labels = []
documents = []
grouped_labels = pd.read_json(dataset_file).groupby("mbti_result")
for label, frame in grouped_labels:
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

freq_on_labels = {}
for i, label in enumerate(labels):
    print(f"{label} popular words are : ")
    print(list(feature_names[popular_words_indices[i]]))
    freq_on_labels[label] = list(feature_names[popular_words_indices[i]])

df = pd.DataFrame.from_dict(freq_on_labels, orient="index")
df.to_csv(f"{stats_dir}/tf-idf.csv", encoding="utf-8")
