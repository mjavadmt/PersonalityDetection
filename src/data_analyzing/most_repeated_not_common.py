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
for label, frame in grouped_labels:
    document_plain_text = " ".join(list(frame["tweets"].apply(lambda x: " ".join(x))))
    frequencies[label] = FreqDist(document_plain_text.split()).most_common()

mmd = 12

