import pandas as pd
from pathlib import Path
import numpy as np
import json
import torch.nn.functional as F
import torch
from transformers import BertModel, BertTokenizerFast

token_counter = 1
tokens = {"UNK": 0}
training_size = 400
dimension = 768


def word_length_on_tweets(tweets: list):
    words_length = np.array([len(i) for i in " ".join(tweets).split()][:dimension])
    if len(words_length) < dimension:
        words_length = np.pad(words_length, (0, dimension - len(words_length)), 'constant', constant_values=0)
    return words_length


def extract_sentence_length(df: pd.DataFrame):
    return df["tweets"].apply(lambda x: len(x))


def extract_word_length(df: pd.DataFrame):
    return df["tweets"].apply(word_length_on_tweets)


def add_to_tokens(tweets: list):
    global token_counter, tokens
    for tweet in tweets:
        for word in tweet.split():
            exists = tokens.get(word, -1)
            if exists == -1:  # doesn't exist
                tokens[word] = token_counter
                token_counter += 1


def assign_tokens(df: pd.DataFrame):
    df["tweets"].apply(lambda x: add_to_tokens(x))


def select_tokens(tweets: list):
    unique_nums = []
    stop_criterion = False
    for tweet in tweets:
        for word in tweet.split():
            word_num = tokens.get(word, -1)
            if word_num == -1:
                word_num = tokens["UNK"]
            unique_nums.append(word_num)
            if len(unique_nums) == dimension:
                stop_criterion = True
                break
        if stop_criterion:
            break
    if len(unique_nums) < dimension:
        for i in range(dimension - len(unique_nums)):
            unique_nums.append(0)
    return np.array(unique_nums)


def words_uni_gram(df: pd.DataFrame):
    unique_nums = df["tweets"].apply(select_tokens)
    return unique_nums


def select_bi_gram_tokens(tweets: list):
    bi_gram_nums = []
    stop_criterion = False
    for tweet in tweets:
        words = tweet.split()
        for i in range(0, len(words) - 1):
            word_num_first = tokens.get(words[i], -1)
            if word_num_first == -1:
                word_num_first = tokens["UNK"]
            word_num_second = tokens.get(words[i + 1], -1)
            if word_num_second == -1:
                word_num_second = tokens["UNK"]
            concatenated_feature = int(f"{word_num_first}{word_num_second}")
            bi_gram_nums.append(concatenated_feature)
            if len(bi_gram_nums) == dimension:
                stop_criterion = True
                break
        if stop_criterion:
            break
    if len(bi_gram_nums) < dimension:
        for i in range(dimension - len(bi_gram_nums)):
            bi_gram_nums.append(0)
    return np.array(bi_gram_nums)


def words_bi_gram(df: pd.DataFrame):
    bi_gram_nums = df["tweets"].apply(select_bi_gram_tokens)
    return bi_gram_nums


def extract_vector(vectors, tokens_w2v, query_word, log_file):
    word_exists = tokens_w2v.get(query_word, -1)
    if word_exists == -1:
        # print(f"the qurey word {query_word} doesn't exist")
        with open(log_file.as_posix(), "a+", encoding="utf-8") as f:
            f.write("---------------\n")
            f.write(
                f"the query word {query_word} to get vector from doesn't exist\n")
            f.write("the error occured in src/word2vec/query.py file\n")
        return False
    else:
        word_vector = vectors[tokens_w2v[query_word]]
        # print(f"word vector of {query_word} is : {vectors[tokens_w2v[query_word]]}")
        return word_vector


def extract_w2v(tweets, vectors, tokens_word2vec, log_file):
    curr_vector = np.zeros(10)
    top_k = 50
    for tweet in tweets[:top_k]:
        for word in tweet.split():
            word_vector = extract_vector(vectors, tokens_word2vec, word, log_file)
            if word_vector is not False:
                curr_vector += word_vector
    curr_vector = np.pad(curr_vector, (0, dimension - 10), 'constant', constant_values=0)
    return curr_vector


def load_word2vec_model():
    root_dir = Path(__file__).resolve().parents[2]
    model_file = root_dir / "models" / "all.word2vec.npy"
    token_file = root_dir / "experiments" / "word2vec" / "all_tokens.json"
    logs_dir = root_dir / "logs"
    log_file = logs_dir / "word2vec.log"

    vectors = np.load(model_file.as_posix())
    with open(token_file.as_posix(), "r") as f:
        tokens_word2vec = json.load(f)

    return vectors, tokens_word2vec, log_file


def extract_w2v_bi_gram(tweets, vectors, tokens_word2vec, log_file):
    curr_vector = np.zeros(20)
    top_k = 50
    for tweet in tweets[:top_k]:
        words = tweet.split()
        for i in range(0, len(words) - 1):
            first_vector = extract_vector(vectors, tokens_word2vec, words[i], log_file)
            if first_vector is False:
                first_vector = np.zeros(10)
            second_vector = extract_vector(vectors, tokens_word2vec, words[i + 1], log_file)
            if second_vector is False:
                second_vector = np.zeros(10)
            curr_vector += np.concatenate((first_vector, second_vector))
    curr_vector = np.pad(curr_vector, (0, dimension - 20), 'constant', constant_values=0)
    return curr_vector


def word2vec_feature(df: pd.DataFrame):
    vectors, tokens_word2vec, log_file = load_word2vec_model()
    return df["tweets"].apply(lambda x: extract_w2v(x, vectors, tokens_word2vec, log_file))


def word2vec_bi_gram(df: pd.DataFrame):
    vectors, tokens_word2vec, log_file = load_word2vec_model()
    return df["tweets"].apply(lambda x: extract_w2v_bi_gram(x, vectors, tokens_word2vec, log_file))


def tweets_embedding(tweets, tokenizer, embeddings_dict, dimension):
    top_k = 128
    tokenized_input = tokenizer.batch_encode_plus(tweets[:top_k], add_special_tokens=True, truncation=True,
                                                  padding=True)
    input_ids_batch = torch.tensor(tokenized_input["input_ids"])
    embeddings = embeddings_dict[input_ids_batch]
    tweet_sum = torch.sum(embeddings, axis=1)
    total_sum = torch.sum(tweet_sum, axis=0)

    return total_sum


def pars_bert_feature(df: pd.DataFrame):
    tokenizer = BertTokenizerFast.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    bert = BertModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    embedding_matrix = bert.embeddings.word_embeddings.weight
    feature_dimension = embedding_matrix.shape[1]
    # extract this tensors from pytorch and convert it to numpy
    return torch.stack(
        df["tweets"].apply(lambda x: tweets_embedding(x, tokenizer, embedding_matrix, feature_dimension)).tolist())
