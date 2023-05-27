# PersonalityDetection

This repository is for personality detection task based Myers-Briggs(MBTI) 16 types on [Persian dataset](https://huggingface.co/datasets/mjavadmt/mbti-persian-twitter). 
You can simply do the data collection and data cleaning and tokenizing task by running *run.bat all* on root folder.
- remember to install `requirements.txt` with running command *pip install -r requirements.txt*

there are various subparts of this module that each will be done individually instead of *all* args:
- *collect* : do just the crawling part
- *clean* : do just the cleaning part
- *tokenize* : do just the tokenizing part
- *analyze* : do just analyze part
-- remember that dataset created during running of code is sample of our whole dataset. for analyzing part you should download
dataset from this [Link](https://huggingface.co/datasets/mjavadmt/mbti-persian-twitter). after downloading you should extract and locate the file in folder data/clean/ with name main_datasets.json

## Data Collection
this part is done by searching for MBTI keywords on each user and then collecting all of this user's tweets(with *tweepy*).

## Data Cleaning
this part has been done by *F#* programming which is faster than *pandas* and other *python* library.
this task is mainly doing the data cleaning on our Persian dataset. 
Operation of this code : 
- removing any non-Persian character
- removing emoji
- removing links and usernames 

at the end we've got *aggressively_cleaned_dataset* which are each users' tweets separated just with space(for this project
we are using this type of cleaning).
after finishing code we extracted an executable file with name *ptpd.exe* cleaner which receives raw json file and the clean that.

## Tokenizer
currently we're using *hazm* library to tokenize sentences and words

## Data Analyzing
there is seperate folder for analyzing data which has various kinds of analyzing for example:
- calculating number of words, sentences, unique words
- most common words for each label 
- tf-idf critertion for each label
- relative normalized frequency for each 2 label pair
- and other ...