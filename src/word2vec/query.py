import json
import numpy as np
from pathlib import Path
from numpy.linalg import norm


def extract_vector(vectors, tokens, query_word, log_file):
    word_exists = tokens.get(query_word, -1)
    if word_exists == -1:
        print(f"the query word {query_word} doesn't exist")
        with open(log_file.as_posix(), "a+", encoding="utf-8") as f:
            f.write("---------------\n")
            f.write(
                f"the query word {query_word} to get vector from doesn't exist\n")
            f.write("the error occurred in src/word2vec/query.py file\n")
        return False
    else:
        word_vector = vectors[tokens[query_word]]
        print(f"word vector of {query_word} is : {vectors[tokens[query_word]]}")
        return word_vector


def check_works():
    traits = [["I", "E"], ["S", "N"], ["T", "F"], ["J", "P"]]
    trait = 0

    root_dir = Path(__file__).resolve().parents[2]
    model_file = root_dir / "models" / f"{traits[trait][1]}_{trait}.word2vec.npy"
    token_file = root_dir / "experiments" / "word2vec" / \
                 f"{traits[trait][0]}_{trait}_tokens.json"
    logs_dir = root_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "word2vec.log"
    vectors = np.load(model_file.as_posix())
    tokens = {}
    with open(token_file.as_posix(), "r") as f:
        tokens = json.load(f)

    query_word = "میسشیسش"

    extract_vector(vectors, tokens, query_word, log_file)

# check_works()
