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
labels_word_count = {}
for label, frame in grouped_labels:
    print(f"gathering for {label}")
    labels.append(label)
    document_plain_text = " ".join(list(frame["tweets"].apply(lambda x: " ".join(x))))
    splitted = document_plain_text.split()
    labels_word_count[label] = len(splitted)
    label_freq = FreqDist(splitted)
    label_freq_sorted = label_freq.most_common()
    frequencies[label] = [label_freq, label_freq_sorted]

df = pd.DataFrame(index=labels, columns=labels)
for i in range(16):
    curr_label = labels[i]
    most_frequent = frequencies[curr_label][1][:10]
    for j in range(16):
        if i == j:
            continue
        other_label = labels[j]
        labels_relative_freq = []
        for word, frequency in most_frequent:
            other_label_word_count = frequencies[other_label][0][word]
            upside = frequency / labels_word_count[curr_label]
            downside = other_label_word_count / labels_word_count[other_label]
            relative_freq = upside / downside
            labels_relative_freq.append((word, round(relative_freq, 2)))
        df.loc[curr_label, other_label] = labels_relative_freq

df.to_json(f"{stats_dir}/RNF.json", index=True)
