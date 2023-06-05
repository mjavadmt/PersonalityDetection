import pandas as pd
from pathlib import Path
import numpy as np
from nltk import FreqDist


root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()

documents = []
grouped_labels = pd.read_json(dataset_file).groupby("mbti_result")
frequencies = {}
labels = []
for label, frame in grouped_labels:
    print(f"gathering for {label}")
    labels.append(label)
    document_plain_text = " ".join(list(frame["tweets"].apply(lambda x: " ".join(x))))
    splitted = document_plain_text.split()
    label_freq = FreqDist(splitted)
    frequencies[label] = label_freq

common = pd.DataFrame(index=labels, columns=labels)
not_common = pd.DataFrame(index=labels, columns=labels)

for i in range(16):
    curr_label = labels[i]
    curr_label_words = set(frequencies[curr_label])
    for j in range(16):
        if i == j:
            continue
        other_label = labels[j]
        other_label_words = set(frequencies[other_label])
        common.loc[curr_label, other_label] = len(curr_label_words.intersection(other_label_words))
        not_common.loc[curr_label, other_label] = len(curr_label_words.difference(other_label_words))

common.to_csv(f"{stats_dir}/common_count.csv", index=True)
not_common.to_csv(f"{stats_dir}/not_common_count.csv", index=True)
