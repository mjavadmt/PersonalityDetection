from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from matplotlib.pyplot import figure
import pandas as pd

root_dir = Path(__file__).resolve().parents[2]
files_dir = root_dir / "stats" / "model_architecture" / "bert_into_bert"

for i, f_name in enumerate(os.listdir(files_dir)):
    df = pd.read_csv(f"{files_dir}/{f_name}", index_col=[0])
    df["acc"] = df["acc"].round(2)
    df["loss"] = df["loss"].round(2)
    df.index = ["train", "validation", "test"]
    df.to_csv(f"{files_dir}/{f_name}")