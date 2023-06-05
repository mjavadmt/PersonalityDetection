import pandas as pd
from pathlib import Path
import numpy as np
from nltk import FreqDist


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

final_df = pd.DataFrame()
for trait in range(4):
    curr_label_str = f"trait_{trait}"
    curr_df = dataframe[["tweets", curr_label_str]]
    grouped_curr_df = curr_df.groupby(curr_label_str)
    documents = []
    frequencies = {}
    labels = []
    for label, frame in grouped_curr_df:
        print(f"gathering for trait {label}")
        labels.append(label)
        document_plain_text = " ".join(list(frame["tweets"].apply(lambda x: " ".join(x))))
        splitted = document_plain_text.split()
        label_freq = FreqDist(splitted)
        frequencies[label] = label_freq

    for i in range(2):
        curr_label = labels[i]
        curr_label_words = set(frequencies[curr_label])
        for j in range(2):
            if i == j:
                continue
            other_label = labels[j]
            other_label_words = set(frequencies[other_label])
            final_df.loc[f"{curr_label}-{other_label}", "intersect"] = len(
                curr_label_words.intersection(other_label_words))
            final_df.loc[f"{curr_label}-{other_label}", "difference"] = len(
                curr_label_words.difference(other_label_words))
final_df.astype(int).to_csv(f"{stats_dir}/common_not_common_pair.csv")
