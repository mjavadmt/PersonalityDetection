import sentencepiece as spm
from sklearn.model_selection import KFold
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()

df = pd.read_json(dataset_file.as_posix())
num_split = 5
top_tweets = 30
df["tweets"] = df["tweets"].apply(lambda x: x[:top_tweets])
tweets = df["tweets"]
kf = KFold(n_splits=num_split)

sizes = [1000, 2000, 3000, 4000]


def extract_tokens(user_tweets, tokens):
    for tweet in user_tweets:
        words = set(tweet.split())
        tokens.update(words)


unknown_to_all = {}
for vocab_size in sizes:
    unknown_to_all[f"vocab-size-{vocab_size}"] = {}
    for fold in range(1, 6):
        unknown_to_all[f"vocab-size-{vocab_size}"][f"fold-{fold}"] = {}

for vocab_size in sizes:
    counter = 1
    print(f"working on vocab size {vocab_size}")
    for train_index, test_index in kf.split(tweets):
        print(f"working on {counter} fold ...")
        print()
        tokens = set()
        sentences = tweets.iloc[train_index]
        sentences.apply(lambda x: extract_tokens(x, tokens))
        unknown_tokens = 0
        total_tokens = len(tokens)
        model_path = root_dir / "models" / f"fold_{counter}_v_{vocab_size}.model"
        model_path = model_path.as_posix()
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        for token in tokens:
            token_prefixed = "▁" + token
            if not sp.PieceToId(token) and not sp.PieceToId(token_prefixed):
                unknown_tokens += 1
        unknown_to_all[f"vocab-size-{vocab_size}"][f"fold-{counter}"] = ((unknown_tokens // 10) / total_tokens) * 100
        counter += 1
    print()

df_stat = pd.DataFrame(unknown_to_all)
for vocab_size in sizes:
    df_stat.loc["all-average", f"vocab-size-{vocab_size}"] = df_stat[f"vocab-size-{vocab_size}"].mean()
df_stat.to_csv(f"{stats_dir}/unk-percentage.csv")
plt.scatter(sizes, df_stat.loc["all-average"])
plt.title("ratio of unknown token to all")
plt.xlabel("vocab sizes")
plt.ylabel("ratio of unk")
plt.savefig(f"{stats_dir}/unk-percentage.png")

