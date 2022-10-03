from sklearn.model_selection import train_test_split
import pandas as pd
import pdb

manifest = pd.read_csv(f"/shared/mandi/all_rlbench_datamanifest.csv")

txt_to_idx = {}
for idx, row in manifest.iterrows():
    txt = row.txt
    if txt not in txt_to_idx.keys():
        txt_to_idx[txt] = []
    txt_to_idx[txt].append(idx)
tasks = txt_to_idx.keys()
tasks_df = pd.DataFrame(tasks)
tasks_df.to_csv('/shared/ademi_adeniji/r3m/tasks.csv')
