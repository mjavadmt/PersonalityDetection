import pandas as pd
from pathlib import Path
from hazm import sent_tokenize, word_tokenize

root_dir = Path(__file__).resolve().parents[3]
final_dataset_path = root_dir / "data" / "clean" / "datasets.json"
final_dataset_path = final_dataset_path.as_posix()
final_dataset = pd.read_json(final_dataset_path)


def make_dir(name):
    path = root_dir / "data" / f"{name}broken"
    path.mkdir(parents=True, exist_ok=True)
    path = path.as_posix()
    return path


def tokenize_a_user_sentences(row):
    tweets = pd.Series(row["tweets"])
    tokenized_sentences = tweets.apply(lambda x: sent_tokenize(x))
    row["tokenized_sentences"] = list(tokenized_sentences)
    return row


def tokenize_a_user_words(row):
    tweets = pd.Series(row["tweets"])
    tokenized_words = tweets.apply(lambda x: word_tokenize(x))
    row["tokenized_words"] = list(tokenized_words)
    return row


def sentence_tokenizer():
    print("sentence tokenizer ...")
    df = final_dataset.apply(tokenize_a_user_sentences, axis=1)
    path = make_dir("sentence")
    df.to_json(f"{path}/sent_tokenized.json")


def word_tokenizer():
    print("word tokenizer ...")
    df = final_dataset.apply(tokenize_a_user_words, axis=1)
    path = make_dir("word")
    df.to_json(f"{path}/word_tokenized.json")


# sentence_tokenizer()
