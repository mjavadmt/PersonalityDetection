import pandas as pd
from data_collection.collect_keyword_tweets import retrieve_tweets
from data_collection.specific_user_tweets import check_and_collect
from pathlib import Path
import os

curr_dir = Path(__file__).resolve().parents[0]
root_dir = Path(__file__).resolve().parents[3]
raw_dir = root_dir / "data" / "raw"
raw_file = raw_dir / "datasets.json"
twint_dir = curr_dir / "output"
twint_file = twint_dir / "user_keywords.csv"

hereee = 1222

usernames_crawled = set()


def collect_specific_user_tweets(row):
    if row["username"] not in usernames_crawled:
        tweets, bio = check_and_collect(row["username"])
        row["tweets"] = tweets
        row["bio"] = bio
        usernames_crawled.add(row["username"])
        return row
    else:
        row["tweets"] = None
        row["bio"] = None
        return row


def main_collector():
    retrieve_tweets()  # after this a csv file will be generated
    df = pd.read_csv(twint_file.as_posix())
    df = df.iloc[:20, :]  # this line should be commented
    df = df[["username", "keyword"]]
    df = df.rename(columns={"keyword": "mbti_label"})
    df = df.apply(collect_specific_user_tweets, axis=1)
    df = df[~df["tweets"].isna()]
    df.to_json(raw_file.as_posix())  # save file in raw

# main_collector()
