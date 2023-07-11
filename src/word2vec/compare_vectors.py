import json
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from query import extract_vector


def main():
    traits = [["I", "E"], ["S", "N"], ["T", "F"], ["J", "P"]]
    trait = 0

    root_dir = Path(__file__).resolve().parents[2]
    model_file_0 = root_dir / "models" / \
                   f"{traits[trait][0]}_{trait}.word2vec.npy"
    model_file_1 = root_dir / "models" / \
                   f"{traits[trait][1]}_{trait}.word2vec.npy"
    token_file_0 = root_dir / "experiments" / "word2vec" / \
                   f"{traits[trait][0]}_{trait}_tokens.json"
    token_file_1 = root_dir / "experiments" / "word2vec" / \
                   f"{traits[trait][1]}_{trait}_tokens.json"
    logs_dir = root_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "word2vec.log"

    tokens_0 = {}
    tokens_1 = {}

    vectors_0 = np.load(model_file_0.as_posix())
    vectors_1 = np.load(model_file_1.as_posix())
    with open(token_file_0.as_posix(), "r") as f:
        tokens_0 = json.load(f)

    with open(token_file_1.as_posix(), "r") as f:
        tokens_1 = json.load(f)

    query_word = "شما"
    print("In I trait word vectors is : ")
    vector_0 = extract_vector(vectors_0, tokens_0, query_word, log_file)
    print()
    print("In E trait word vector is")
    vector_1 = extract_vector(vectors_1, tokens_1, query_word, log_file)
    print()
    if vector_0 is False or vector_1 is False:
        print("vectors doesnt exist in at least one model")
        return
    else:
        cos_sim = round(np.dot(vector_0, vector_1) /
                        (norm(vector_0) * norm(vector_1)), 5)
        print(f"cosine similarity between vectors of two model is {cos_sim}")


main()
