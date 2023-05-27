import logging
import os
import threading
from pathlib import Path
import json

import tweepy
import pandas as pd

from datetime import timedelta, datetime

project_dir = Path(__file__).resolve().parents[3]
data_dir = project_dir / 'data' / 'raw'
curr_dir = Path(__file__).resolve().parents[0].as_posix()


class TweepySingleton:
    output_dir = data_dir / 'crawled'
    failed_users = data_dir / 'failed.txt'
    bios_dir = output_dir / 'bios'
    tweets_dir = output_dir / 'tweets'

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(TweepySingleton, cls).__new__(cls)
                    cls._instance.__setup()
        return cls._instance

    def __setup(self):
        self.api = self._create_api()
        self.logger = logging.getLogger(__name__)

        self.tweets_dir.mkdir(parents=True, exist_ok=True)
        self.bios_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _create_api():
        f = open(f"{curr_dir}/secrets.json")
        api_tokens = json.load(f)
        auth = tweepy.OAuthHandler(
            api_tokens["consumer_key"], api_tokens["consumer_secret"])
        auth.set_access_token(
            api_tokens["access_token"], api_tokens["access_token_secret"])
        return tweepy.API(auth, wait_on_rate_limit=True)

    def get_user_info(self, handle: str):
        try:
            user: tweepy.User = self.api.get_user(screen_name=handle)
        except Exception as e:
            print("----------------")
            print(f"{handle} got error")
            print(e)
            print("----------------")

        return {
            'tweet_count': user.statuses_count,
            'protected': user.protected,
            'followers_count': user.followers_count,
            'friends_count': user.friends_count,
            'created_at': user.created_at.isoformat()
        }

    def get_user_tweets(self, handle: str, count=100):
        tweets = []
        print(f"trying for user {handle}")
        self.logger.info(f'getting user object of {handle}')
        try:
            user = self.api.get_user(screen_name=handle)

            if user.protected or not hasattr(user, 'status'):
                msg = f'user is protected or has no tweets'
                self.logger.error(msg)
                with open(self.failed_users.as_posix(), 'a') as f:
                    f.write(f"{handle}\t{msg}\n")

            self.logger.info(f'getting bio of {handle}')

            with open((self.bios_dir / f"{handle}.txt").as_posix(), 'w', encoding='utf-8') as f:
                f.write(user.description)

            self.logger.info(f'getting tweets of {handle}')

            for tweet in tweepy.Cursor(self.api.user_timeline,
                                       screen_name=handle,
                                       tweet_mode='extended',
                                       include_rts=False,
                                       ).items(count):
                tweets.append(tweet.full_text)

        except tweepy.TweepyException as e:
            self.logger.error(f'error getting user {handle} \n -- \n {e}')

            with open(self.failed_users.as_posix(), 'a') as f:
                f.write(f"{handle}\tTweepy Error\n")
            return None, None

        df = pd.DataFrame(tweets, columns=['tweet'])
        df.to_json(
            (self.tweets_dir / f"{handle}.json").as_posix(), indent=2, orient='records')
        self.logger.info(f"saved tweets of {handle}")

        return tweets, user.description


def user_is_not_valid(user_info):
    return user_info["tweet_count"] < 150 or user_info["protected"] or user_info["followers_count"] < 5 or user_info[
        "created_at"] > (datetime.now() - timedelta(days=30)).isoformat()


def check_and_collect(username):
    tweepy_obj = TweepySingleton()
    try:
        user_detail = tweepy_obj.get_user_info(username)
        if not user_is_not_valid(user_detail):
            return tweepy_obj.get_user_tweets(username, 10)  # this num should be user_detail["tweet_count"]
    except:
        return None, None
    return None, None


# check_and_collect("neutronofh")
