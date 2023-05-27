import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
dataset_file = root_dir / "data" / "clean" / "main_datasets.json"
stats_dir = (root_dir / "stats").as_posix()

df_0 = pd.read_csv(f"{stats_dir}/common_count.csv", index_col=0)
df_1 = pd.read_csv(f"{stats_dir}/not_common_count.csv", index_col=0)

df_0_agg = df_0.agg(["min", "max"], axis=0).T.astype(int)
df_1_agg = df_1.agg(["min", "max"], axis=0).T.astype(int)
plt.figure(figsize=(12, 5))
bar1 = plt.bar(np.arange(len(df_0_agg)) + 0.4, df_0_agg["min"], tick_label=df_0_agg.index, alpha=0.4, color='r',
               label='min', align='edge', width=0.4)
bar2 = plt.bar(range(len(df_0_agg)), df_0_agg["max"], tick_label=df_0_agg.index, alpha=0.4, color='b',
               label='max', align='edge', width=0.4)
plt.title(f"extract min and max number of common between label pairs", fontweight="bold")
plt.xticks(rotation=30)
indexes = list(df_0.idxmin(axis=1)) + list(df_0.idxmax(axis=1))
i = 0
for rect in bar1 + bar2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, indexes[i], ha='center', va='bottom')
    i += 1
plt.legend()
plt.draw()

plt.figure(figsize=(12, 5))
bar3 = plt.bar(np.arange(len(df_1_agg)) + 0.4, df_1_agg["min"], tick_label=df_1_agg.index, alpha=0.4, color='r',
               label='min', align='edge', width=0.4)
bar4 = plt.bar(range(len(df_1_agg)), df_1_agg["max"], tick_label=df_1_agg.index, alpha=0.4, color='b',
               label='max', align='edge', width=0.4)
plt.title(f"extract min and max number of not common between label pairs", fontweight="bold")
plt.xticks(rotation=30)
indexes = list(df_1.idxmin(axis=1)) + list(df_1.idxmax(axis=1))
i = 0
for rect in bar3 + bar4:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, indexes[i], ha='center', va='bottom')
    i += 1
plt.legend()
plt.draw()

