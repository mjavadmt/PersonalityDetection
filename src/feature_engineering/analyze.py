import json
from pathlib import Path
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parents[2]
experiment_feat_engineering_dir = root_dir / "experiments" / "feat_engineering"
stats_dir = (root_dir / "stats").as_posix()

with open(f"{experiment_feat_engineering_dir.as_posix()}/results.json", "r") as f:
    results = json.load(f)

acc = results["acc"]
loss = results["loss"]

counter = 0
for key_0, value_0 in acc.items():
    if counter % 2 == 0 and 0 < counter <= 4:
        plt.savefig(f"{stats_dir}/acc_feat_engineering_{counter // 2}.png")
        plt.figure()

    counter += 1
    for key_1, value_1 in value_0.items():
        plt.plot(value_1, label=f"{key_0}-{key_1}")
        plt.title("accuracy based on feature")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
plt.savefig(f"{stats_dir}/acc_feat_engineering_{counter // 2}.png")
plt.figure()
counter = 0
for key_0, value_0 in loss.items():
    if counter % 2 == 0 and 0 < counter <= 4:
        plt.savefig(f"{stats_dir}/loss_feat_engineering_{counter // 2}.png")
        plt.figure()
    counter += 1
    for key_1, value_1 in value_0.items():
        plt.plot(value_1, label=f"{key_0}-{key_1}")
        plt.title("loss based on feature")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
plt.savefig(f"{stats_dir}/loss_feat_engineering_{counter // 2}.png")
# plt.show()
