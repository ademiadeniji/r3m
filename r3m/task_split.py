from sklearn.model_selection import train_test_split
import pandas as pd
import pdb

# manifest = pd.read_csv(f"/shared/ademi_adeniji/r3m/all_rlbench_datamanifest.csv")
manifest = pd.read_csv(f"/shared/ademi_adeniji/r3m/rlbenchmanifestrgb.csv")
print(len(manifest))
# val_tasks = ['open_drawer', 'close_microwave', 'move_hanger', 'put_item_in_drawer', 'slide_cabinet_open']
# train_df = manifest[~manifest['txt'].isin(val_tasks)]
# print(len(train_df))
# val_df = manifest[manifest['txt'].isin(val_tasks)]
# print(len(val_df))
front_var0_df = pd.DataFrame()
for idx, row in manifest.iterrows():
    if 'front_rgb' in row['path'] and 'variation0' in row['path']:
        front_var0_df = front_var0_df.append(row)

txt_to_idx = {}
for idx, row in front_var0_df.iterrows():
    txt = row.txt
    if txt not in txt_to_idx.keys():
        txt_to_idx[txt] = []
    txt_to_idx[txt].append(idx)

front_var0_df.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbfrontvar0.csv')  
# train_df.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbtraintasks.csv')  
# val_df.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbevaltasks.csv')  