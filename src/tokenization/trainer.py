from sklearn.model_selection import KFold
from pathlib import Path
import pandas as pd
import sentencepiece as spm
from datetime import datetime

root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
log_file = root_dir / "run.log"

df = pd.read_json(dataset_file.as_posix())
num_split = 5
top_tweets = 30
df["tweets"] = df["tweets"].apply(lambda x: x[:top_tweets])
kf = KFold(n_splits=num_split)
tweets = df["tweets"]
sizes = [1000, 2000, 3000, 4000][::-1]

for vocab_size in sizes:
    counter = 1
    print(f"vocab size : {vocab_size}")
    for train_index, test_index in kf.split(tweets):
        print(f"fold : {counter}")
        sentences = tweets.iloc[train_index]
        sentences = list(sentences.apply(lambda x: (" ".join(x)) + "."))
        model_path = root_dir / "models" / f"fold_{counter}_v_{vocab_size}"
        model_path = model_path.as_posix()
        spm.SentencePieceTrainer.Train(sentence_iterator=iter(
            sentences), model_prefix=model_path, vocab_size=vocab_size)

        with open(log_file.as_posix(), "a+") as f:
            f.write("-----------------------------\n")
            curr_datetime = str(datetime.now())
            f.write(f"at time : {curr_datetime[:-7]}\n")
            f.write(
                f"tokenizer on k fold : {counter} and vocab size : {vocab_size} has been trained and saved "
                f"successfully\n")
        counter += 1
