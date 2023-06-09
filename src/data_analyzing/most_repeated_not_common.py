import pandas as pd
from pathlib import Path
import numpy as np
from nltk import FreqDist
import persian



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
    label_freq_sorted = label_freq.most_common()
    frequencies[label] = [label_freq, label_freq_sorted]

df = pd.DataFrame(index=labels, columns=labels)
for i in range(16):
    curr_label = labels[i]
    most_frequent = frequencies[curr_label][1]
    for j in range(16):
        if i == j:
            continue
        found_not_common = []
        other_label = labels[j]
        for word, frequency in most_frequent:
            persian_word = persian.convert_ar_characters(word)
            count_in_other_label_1 = frequencies[other_label][0][word]
            count_in_other_label_2 = frequencies[other_label][0][persian_word]
            if count_in_other_label_1 == 0 and count_in_other_label_2 == 0:
                found_not_common.append(word)
                print(f"{curr_label} is finding for {other_label} : found {len(found_not_common)}")
                if len(found_not_common) == 10:
                    break
        df.loc[curr_label, other_label] = found_not_common

df.to_json(f"{stats_dir}/most_not_common.json", index=True)
