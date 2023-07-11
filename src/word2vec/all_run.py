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





root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
experiment_word2vec_dir = root_dir / "experiments" / "word2vec"
log_file = root_dir / "run.log"
models_dir = root_dir / "models"

dataframe = pd.read_json(dataset_file)
top_k = 200

model_file = f"all.word2vec.npy"
model_file = models_dir / model_file
if model_file.exists():
    print(f"model 'all' exists :)")
else:
    sentences = list(dataframe["tweets"].iloc[:top_k].apply(
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
        f.write(f"model word2vec 'all' is done\n")
    np.save(f"{models_dir.as_posix()}/all.word2vec.npy", wordVectors)
    with open(f"{experiment_word2vec_dir.as_posix()}/all_tokens.json", "w") as f:
        json.dump(tokens, f)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("training took %d seconds" % (time.time() - startTime))
