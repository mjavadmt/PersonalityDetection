from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from matplotlib.pyplot import figure

root_dir = Path(__file__).resolve().parents[2]
files_dir = root_dir / "stats" / "model_architecture" / "bert_truncated"

traits = ["I/E", "S/N", "T/F", "P/J"]


for i, f_name in enumerate(os.listdir(files_dir)):
    if not f_name.endswith(".json"):
        continue
    with open(f"{files_dir.as_posix()}/{f_name}", "r") as f:
        figure(figsize=(6, 6), dpi=80)
        results = json.load(f)
        acc = results["acc"]
        loss = results["loss"]
        plt.subplot(1, 2, 1)
        for key, value in acc.items():
            plt.plot(value, label=key)
        plt.title(f"accuracy trait {traits[i]}")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        for key, value in loss.items():
            plt.plot(value, label=key)
        plt.title(f"loss trait {traits[i]}")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"{files_dir.as_posix()}/trait_{i}.png")
        if i == 3:
            continue
        plt.figure()
