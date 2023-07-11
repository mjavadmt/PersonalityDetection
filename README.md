# PersonalityDetection

This repository is for personality detection task based Myers-Briggs(MBTI) 16 types on [Persian dataset](https://huggingface.co/datasets/mjavadmt/mbti-persian-twitter). This project has been implemented in 2 phase.
- remember to install `requirements.txt` with running command **pip install -r requirements.txt**
## Phase 1
This phase is mainly performing tasks on data collecting and cleaning.
You can simply do the data collection and data cleaning and tokenizing task by running ***run_1.bat all*** on root folder.

there are various subparts of this module that each will be done individually instead of *all* args:
- *collect* : do just the crawling part
- *clean* : do just the cleaning part
- *tokenize* : do just the tokenizing part
- *analyze* : do just analyze part
- *report* : making report of this project

-- remember that dataset created during running of code is sample of our whole dataset. for analyzing part you should download
**dataset from this [Link](https://huggingface.co/datasets/mjavadmt/mbti-persian-twitter). after downloading you should extract and locate the file in folder data/clean/ with name main_datasets.json**

### Data Collection
this part is done by searching for MBTI keywords on each user and then collecting all of this user's tweets(with *tweepy*).

### Data Cleaning
this part has been done by *F#* programming which is faster than *pandas* and other *python* library.
this task is mainly doing the data cleaning on our Persian dataset. 
Operation of this code : 
- removing any non-Persian character
- removing emoji
- removing links and usernames 

at the end we've got *aggressively_cleaned_dataset* which are each users' tweets separated just with space(for this project
we are using this type of cleaning).
after finishing code we extracted an executable file with name *ptpd.exe* cleaner which receives raw json file and the clean that.

### Tokenizer
currently we're using *hazm* library to tokenize sentences and words

### Data Analyzing
there is seperate folder for analyzing data which has various kinds of analyzing for example:
- calculating number of words, sentences, unique words
- most common words for each label 
- tf-idf critertion for each label
- relative normalized frequency for each 2 label pair
- and other ...

## Phase 2
This Phase is concentrating on developing differnet models. Various experminets has been implemented to evaluate and retreive final best model.
<br/>
Like former phase you can run all parts by executing ***run_2.bat all*** 
<br/>
-- some parts couldn't be automated so that files are provided in python notebook to execute manually(lack of enough gpu on local made me do that)

<br/>
The experiments has been made through this phase is divided into following subparts:
- word2vec : this part trains word2vec model with skip-gram approach and negative sampling on each labels separately. it also train on whole dataset and produce `all.word2vec.npy` model 
- tokenization : trains `SentencePiece` model with different size of *vocab_size* on our dataset with implementing KFold cross validation. and then calculates *unk* token on each *vocab_size*.
- language model : this part uses `GPT2-Medium` for fine tuning on each label of our dataset. and then generate sentences for each part which their examples are available in stats folder.
- feature engineering : defines a general NN(Neural Network) model and them examines different features on this model and then deduce which feature will lead to better perfomance on our model.
- model architecture : this part examinates different models to acheive the best accuracy. Implemented models were `combined ParsBERT + word2vec`, `BERT truncated` model which truncate input to be no more than 512 tokens and last model was `BERT into BERT` that use first BERT as feature extractor and then feed embeddings to the second BERT(fine tunes happens on the second BERT)
- data augmentation : uses OpenAI api for feeding and fine tuning our dataset on ChatGPT. use different prompt for creating sample of that label.
- zero shot classification with ChatGPT : feed tweets without their labels to ChatGPT, and then ask ChatGPT to classify between 16 labels. 
  
there are various subparts of this module that each will be done individually instead of *all* args:
- *word2vec* : just runs word2vec
- *tokenization* : just runs tokenization
- *feat_engineering* : just runs feature enginnering
- *model_architecture* : just runs model architecture