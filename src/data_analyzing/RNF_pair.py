import pandas as pd
from pathlib import Path
import numpy as np
from nltk import FreqDist
import persian


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
traits = ["E-I", "I-E", "N-S", "S-N", "F-T", "T-F", "J-P", "P-J"]
indexes = traits
final_df = pd.DataFrame(index=indexes, columns=[i for i in range(10)])

for trait in range(4):
    curr_label_str = f"trait_{trait}"
    curr_df = dataframe[["tweets", curr_label_str]]
    grouped_curr_df = curr_df.groupby(curr_label_str)
    documents = []
    frequencies = {}
    labels = []
    labels_word_count = {}
    for label, frame in grouped_curr_df:
        print(f"gathering for trait {label}")
        labels.append(label)
        document_plain_text = " ".join(list(frame["tweets"].apply(lambda x: " ".join(x))))
        splitted = document_plain_text.split()
        labels_word_count[label] = len(splitted)
        label_freq = FreqDist(splitted)
        label_freq_sorted = label_freq.most_common()
        frequencies[label] = [label_freq, label_freq_sorted]

    for i in range(2):
        curr_label = labels[i]
        most_frequent = frequencies[curr_label][1][:10]
        for j in range(2):
            if i == j:
                continue
            found_not_common = []
            other_label = labels[j]
            labels_relative_freq = []
            for word, frequency in most_frequent:
                other_label_word_count = frequencies[other_label][0][word]
                upside = frequency / labels_word_count[curr_label]
                downside = other_label_word_count / labels_word_count[other_label]
                relative_freq = upside / downside
                labels_relative_freq.append((word, round(relative_freq, 2)))
            for k in range(10):
                final_df.loc[
                    f"{curr_label}-{other_label}", k] = f"{labels_relative_freq[k][0]} {labels_relative_freq[k][1]}"
final_df.to_csv(f"{stats_dir}/RNF-pair.csv", encoding="utf-8")
