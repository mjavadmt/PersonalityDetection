import pandas as pd

df = pd.read_json("../../stats/RNF.json")
mmd = 12
labels = list(df.keys())
for i in range(len(labels)):
    for j in range(len(labels)):
        if i == j:
            continue
        curr_label = labels[i]
        other_label = labels[j]
        print(f"for {curr_label} non common with {other_label} are :")
        print(df.loc[curr_label, other_label])
