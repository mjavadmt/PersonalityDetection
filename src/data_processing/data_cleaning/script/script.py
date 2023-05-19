from pathlib import Path
import os
import pandas as pd
import json

root_dir = Path(__file__).resolve().parents[4]
cleaned_dataset_dir = root_dir / "data" / "raw"
crawled_dir = root_dir / "data" / "raw" / "crawled"
tweets_dir = crawled_dir / "tweets"
cleaned_dir = root_dir / "data" / "clean" / "tweets"
cleaned_dir.mkdir(parents=True, exist_ok=True)
cleaned_dir_parent = cleaned_dir.parents[0].as_posix()
curr_dir = Path(__file__).resolve().parents[0]
ptpd_file = curr_dir / "ptpd.exe"


enumerator = 0


def cleaning_tweets(row):
    global enumerator
    enumerator += 1
    username = row["username"]
    curr_path = tweets_dir.as_posix()
    cleaned_dir_path = cleaned_dir.as_posix()
    file_name = f"{curr_path}/{username}.json"
    os.system(
        f"{ptpd_file.as_posix()} raw-text clean-tweets --input-file {file_name} --output-file {cleaned_dir_path}/{username}.json "
        "--clean aggressive")
    with open(f"{cleaned_dir_path}/{username}.json", "r", encoding="utf-8") as f:
        tweets = json.load(f)
    print(f"user {enumerator} is being cleaned ... ")
    row["tweets"] = tweets
    return row


def main_cleaner():
    df = pd.read_json(f"{cleaned_dataset_dir.as_posix()}/datasets.json")
    df = df.apply(cleaning_tweets, axis=1)
    df.to_json(f"{cleaned_dir_parent}/datasets.json")


# main_cleaner()
