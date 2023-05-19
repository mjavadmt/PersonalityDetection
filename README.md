# PersonalityDetection

This repository is for personality detection task based Myers-Briggs(MBTI) 16 types on [Persian dataset](https://huggingface.co/datasets/mjavadmt/mbti-persian-twitter). 
You can simply do the data collection and data cleaning task by running *python src/data_processing/main.py*

there are various subparts of this module that each will be done individually:
- *main_cleaner.py* : do just the cleaning part
- *main_collector.py* : do just the crawling part
- *main_collector_and_cleaner.py* : do both the crawling and cleaning part
- *main_tokenizer.py* : do just the tokenizing part

## Data Collection
this part is done by searching for MBTI keywords on each user and then collecting all of this user's tweets(with -*tweepy*).

## Data Cleaning
this part has been done by *F#* programming which is faster than *pandas* and other *python* library.
this task is mainly doing the data cleaning on our Persian dataset. 
Operation of this code : 
- removing any non-Persian character
- removing emoji
- removing links and usernames
at the end we've got *aggressively_cleaned_dataset* which are each users' tweets separated just with space