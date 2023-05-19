from data_collection.twint_scraper import TwintScraper
import nest_asyncio
import time
import datetime


def scrape_tweets():
    nest_asyncio.apply()
    t0 = time.time()
    scraper = TwintScraper(base_dir="output", max_fail_nums=5)

    t1 = time.time()
    # collect data from twitter
    scraper.run()
    t2 = time.time()

    # remove extra files include *_resume_file.txt
    scraper.remove_resume_files()

    # merge all csv files
    df = scraper.concat(remove_items=True, save_df=True)
    t3 = time.time()

    print("---------------END OF CRAWL---------------")
    print(f"whole time: {datetime.timedelta(seconds=t3 - t0)}")
    print(f"crawl time: {datetime.timedelta(seconds=t2 - t1)}")
    print("number of data: {}".format(len(df)))
    print("oldest tweet created time: {}".format(min(df["date"])))
    print("newest tweet created time: {}".format(max(df["date"])))
