from sklearn.model_selection import train_test_split
import pandas as pd

manifest = pd.read_csv(f"/shared/ademi_adeniji/r3m/rlbenchmanifestrgb.csv")
print(len(manifest))
train, test = train_test_split(manifest, test_size=0.1)
print(len(train))
print(len(test))
train.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbtrain.csv')  
test.to_csv('/shared/ademi_adeniji/r3m/rlbenchmanifestrgbeval.csv')  
