import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()

documents = []
df = pd.read_json(dataset_file)
text = " ".join(list(df["tweets"].apply(lambda x: " ".join(x))))
sentences = ".".join(list(df["tweets"].apply(lambda x: ".".join(x))))
words = text.split()
words_len = len(words)
sentence_len = len(sentences.split("."))
unique_words_len = len(set(words))
data_len = len(df)

stats_dict = {"data-len": data_len, "sentence-len": sentence_len, "word-len": words_len,
              "unique-words-len": unique_words_len}
df_stats = pd.DataFrame(index=stats_dict.keys())
df_stats["stats"] = stats_dict.values()
df_stats.to_csv(f"{stats_dir}/base_data_stats.csv")

