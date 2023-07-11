#!/usr/bin/env python

from sgd import *
from word2vec import *
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from utils.treebank import PersonalityDataset
import matplotlib
import pandas as pd
import json

matplotlib.use('agg')


# Reset the random seed to make sure that everyone gets the same results
random.seed(314)


def make_trait(row):
    row["trait_0"] = row["mbti_result"][0]
    row["trait_1"] = row["mbti_result"][1]
    row["trait_2"] = row["mbti_result"][2]
    row["trait_3"] = row["mbti_result"][3]
    return row


root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
experiment_word2vec_dir = root_dir / "experiments" / "word2vec"
log_file = root_dir / "run.log"
models_dir = root_dir / "models"

print("make 4 individual trait ...")
dataframe = pd.read_json(dataset_file)
dataframe = dataframe.apply(make_trait, axis=1)
top_k = 200

for trait in range(4):
    curr_label_str = f"trait_{trait}"
    curr_df = dataframe[["tweets", curr_label_str]]
    print("separating each label ...")
    grouped_curr_df = curr_df.groupby(curr_label_str)
    documents = []
    frequencies = {}
    labels = []
    print(f"working on {trait + 1}th trait")
    for label, frame in grouped_curr_df:
        model_file = f"{label}_{trait}.word2vec.npy"
        model_file = models_dir / model_file
        if model_file.exists():
            print(f"model {label} exists :)")
            continue
        print(f"trait label : {label}")
        sentences = list(frame["tweets"].iloc[:top_k].apply(
            lambda tweets: " ".join(tweets).split()))
        dataset = PersonalityDataset(sentences=sentences)
        tokens = dataset.tokens()
        nWords = len(tokens)

        # We are going to train 10-dimensional vectors for this assignment
        dimVectors = 10

        # Context size
        C = 5

        # Reset the random seed to make sure that everyone gets the same results
        random.seed(31415)
        np.random.seed(9265)

        startTime = time.time()
        wordVectors = np.concatenate(
            ((np.random.rand(nWords, dimVectors) - 0.5) /
             dimVectors, np.zeros((nWords, dimVectors))),
            axis=0)
        wordVectors = sgd(
            lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                             negSamplingLossAndGradient),
            wordVectors, 0.3, 60, None, True, PRINT_EVERY=4)

        with open(log_file.as_posix(), "a+") as f:
            f.write("-----------------------------\n")
            curr_datetime = str(datetime.now())
            f.write(f"at time : {curr_datetime[:-7]}\n")
            f.write(f"model word2vec for trait {label} is done\n")
        np.save(f"{models_dir.as_posix()}/{label}_{trait}.word2vec.npy", wordVectors)
        with open(f"{experiment_word2vec_dir.as_posix()}/{label}_{trait}_tokens.json", "w") as f:
            json.dump(tokens, f)
        # Note that normalization is not called here. This is not a bug,
        # normalizing during training loses the notion of length.

        print("training took %d seconds" % (time.time() - startTime))
