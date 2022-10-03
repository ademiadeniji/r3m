from sklearn.model_selection import train_test_split
import pandas as pd
import pdb

train_tasks = ['lamp_on', 'lamp_off', 'open_fridge', 'close_fridge', 'close_box',
'open_box', 'put_tray_in_oven', 'take_tray_out_of_oven', 'meat_on_grill', 'meat_off_grill',
'tv_on', 'tf_off', 'open_jar', 'close_jar', 'open_grill', 'close_grill', 'toilet_seat_up',
'toilet_seat_down', 'insert_usb_in_computer', 'take_usb_out_of_computer']
manifest = pd.read_csv(f"/shared/mandi/all_rlbench_datamanifest_train.csv")
print(len(manifest))
train_df = manifest[~manifest['txt'].isin(val_tasks)]
print(len(train_df))
val_df = manifest[manifest['txt'].isin(val_tasks)]
print(len(val_df))

txt_to_idx = {}
for idx, row in train_df.iterrows():
    txt = row.txt
    if txt not in txt_to_idx.keys():
        txt_to_idx[txt] = []
    txt_to_idx[txt].append(idx)
pdb.set_trace()
# train_df.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbtraintasks.csv')  
# val_df.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbevaltasks.csv')  