import time
import tweepy
import pandas as pd
from pathlib import Path
from os.path import exists
import os

curr_dir = Path(__file__).resolve().parents[0]
output_dir = curr_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

KEYWORDS = ["ENFP", "ENFJ", "ESTJ", "ESFJ", "ESTP", "ESFP", "ENTJ", "ENTP", "INFP", "INTP", "INFJ", "INTJ", "ISFP",
            "ISTP", "ISFJ", "ISTJ"]

api_tokens = {
    "tweet_num": 100,
    "consumer_key": 'Tuemyojw81FZZBPpGmxVR00ID',
    "consumer_secret": 'hZQHuYWMotTEHPQyGgEhLYkXKkCDUaPZuEQIkkrPgIcg9vXEzn',
    "access_token": '1429311052605497345-uUrAQLyxcRTpEfHNTKPdf3UByl5LIL',
    "access_token_secret": 'cA6vzIQkk4sk4xXgX3yLmO3M9lwhXv1d0iBHcly6yYax6'
}

auth = tweepy.OAuthHandler(api_tokens["consumer_key"], api_tokens["consumer_secret"])
auth.set_access_token(api_tokens["access_token"], api_tokens["access_token_secret"])
api = tweepy.API(auth, wait_on_rate_limit=True)

data_dict = {
    "user_alias_name": [],
    "username": [],
    "text": [],
    "keyword": []
}


def retrieve_tweets():
    for keyword in KEYWORDS[:6]:  # the [:6] should be removed
        print(f"Collecting tweets for {keyword} personality.")
        count = 0
        desired_count = 2  # this should be increased to 1000
        tweet_count = 0
        q = f'{keyword} -from:advertising_account -filter:statuses_min:25'
        try:
            for tweet in tweepy.Cursor(api.search_tweets, q=q, lang='fa').items():
                if tweet_count >= desired_count:
                    break
                if count % 100 == 0 and count != 0:
                    print(f"In the {count}th iteration, {tweet_count} tweets have been collected.")
                    time.sleep(60)
                count += 1
                user = api.get_user(user_id=tweet.user.id)
                if user.friends_count >= 50 and user.followers_count >= 50:
                    data_dict['username'].append(tweet.user.screen_name)
                    data_dict['user_alias_name'].append(tweet.user.name)
                    data_dict['text'].append(tweet._json['text'])
                    data_dict['keyword'].append(keyword)
                    tweet_count += 1
        except Exception as e:
            print(f"ran into {e} exception")
            pass

        print("===============================================")

    if exists(f"{output_dir.as_posix()}/user_keywords.csv"):
        existed_df = pd.read_csv(f"{output_dir.as_posix()}/user_keywords.csv")
        data = pd.DataFrame(data_dict)
        new_df = pd.concat([existed_df, data], ignore_index=True)
        new_df.to_csv(f"{output_dir.as_posix()}/user_keywords.csv", index=False)
        mmdd = 122
    else:
        data = pd.DataFrame(data_dict)
        data.to_csv(f"{output_dir.as_posix()}/user_keywords.csv", index=False)

    here = 10


# retrieve_tweets()
