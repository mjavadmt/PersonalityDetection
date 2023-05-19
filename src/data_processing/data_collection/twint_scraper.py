import twint
import os
import pandas as pd


class TwintScraper:
    """"Scraper use twint"""

    # all available fields are:
    # id, conversation_id, created_at, date, time, timezone, user_id, username, name, place, tweet, language,
    # mentions, urls, photos, replies_count, retweets_count, likes_count, hashtags, cashtags, link,
    # retweet, quote_url, video, thumbnail, near, geo, source, user_rt_id, user_rt, retweet_id, reply_to,
    # retweet_date, translate, trans_src, trans_dest

    # list of keywords for search
    KEYWORDS = ['ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'Istj', 'Istp', 'Isfj', 'Isfp',
                'INTJ', 'INTP', 'INFJ', 'INFP', 'Intj', 'Intp',
                'Infj', 'Infp', 'istj', 'istp', 'isfj', 'isfp', 'intj', 'intp', 'infj',
                'infp', 'ESTJ', 'ESTP', 'ESFJ', 'ESFP', 'Estj', 'Estp', 'Esfj', 'Esfp',
                'ENTJ', 'ENTP', 'ENFJ', 'ENFP', 'Entj', 'Entp', 'Enfj', 'Enfp', 'estj',
                'estp', 'esfj', 'esfp', 'entj', 'entp', 'enfj', 'enfp']

    def __init__(self, base_dir, max_fail_nums=15):
        """
        :param base_dir: base directory for save files
        """
        self.config = twint.Config()
        self.config.Hide_output = True
        self.config.Lang = "fa"
        self.config.Store_csv = True
        self.config.Pandas = True
        self.config.Pandas_clean = True
        self.config.Limit = 2000
        self.config.Min_likes = 10
        self.base_dir = base_dir
        self.max_fail_nums = max_fail_nums

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

    def run(self):
        """
        Collect tweets by keywords
        """
        for key in self.KEYWORDS:
            self.config.Search = f"{key} lang:fa"
            self.config.Resume = os.path.join(
                self.base_dir, f"{key}_resume_file.txt")
            self.config.Output = os.path.join(self.base_dir, f"{key}.csv")
            twint.run.Search(self.config)

    def remove_resume_files(self):
        """
        remove all resume files
        """
        for key in self.KEYWORDS:
            fname = f"{key}_resume_file.txt"
            fpath = os.path.join(self.base_dir, fname)
            self.remove_file(fpath)

    def concat(self, remove_items=False, save_df=False):
        """
        Concatenate all csv files

        :param remove_items: remove keyword file or not
        :param save_df: save result dataframe in a csv file
        :return: result dataframe
        """
        all_dfs = []

        for key in self.KEYWORDS:
            try:
                fname = key + ".csv"
                fpath = os.path.join(self.base_dir, fname)
                key_df = pd.read_csv(fpath)
                key_df["keyword"] = key
                all_dfs.append(key_df)
            except Exception as e:
                print(e)

            if remove_items:
                self.remove_file(fpath)

        merged_df = pd.concat(all_dfs)

        if save_df:
            merged_df.to_csv(os.path.join(
                self.base_dir, "user_keywords.csv"), index=False)

        return merged_df

    @staticmethod
    def remove_file(fpath):
        """"
        check file exist and remove
        """
        if os.path.exists(fpath):
            os.remove(fpath)
        else:
            print("{} does not exist".format(fpath))
