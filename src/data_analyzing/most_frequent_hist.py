import pandas as pd
from pathlib import Path
from nltk import FreqDist
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from matplotlib.pyplot import figure



root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()
documents = []
df = pd.read_json(dataset_file)
text = " ".join(list(df["tweets"].apply(lambda x: " ".join(x))))
words = text.split()
print("words are splitted")
words_freq = FreqDist(words)
print("word counter done")
sorted_freq = words_freq.most_common()
print("sorting done")
count = 30
first_words = list(map(lambda x: x[0], sorted_freq[:count]))
first_words_counts = list(map(lambda x: x[1], sorted_freq[:count]))
plt.bar(range(count), first_words_counts, tick_label=[get_display(reshape(label)) for label in first_words])
plt.title(f"most frequent words in tweets", fontweight="bold")
plt.xticks(rotation=90)
last_words = list(map(lambda x: x[0], sorted_freq[-count:]))
last_words_counts = list(map(lambda x: x[1], sorted_freq[-count:]))
plt.draw()
figure(figsize=(8, 7), dpi=60)
plt.bar(range(count), last_words_counts, tick_label=[get_display(reshape(label)) for label in last_words])
plt.title(f"least frequent words in tweets", fontweight="bold")
plt.xticks(rotation=90)
plt.draw()
here = 12
