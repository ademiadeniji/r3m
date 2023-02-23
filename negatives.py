import json
import csv
import os

# create dictionary of placeholders to labels
placeholders_to_labels = {}
# iterate through json file called /shared/ademi_adeniji/something-something/something-something-v2-train.csv
with open('/shared/ademi_adeniji/something-something/something-something-v2-train.json') as json_file:
    data = json.load(json_file)
    for row in data:
        for placeholder in row["placeholders"]:
            if placeholder in placeholders_to_labels:
                if row["template"] not in placeholders_to_labels[placeholder]:
                    placeholders_to_labels[placeholder].append(row["template"])
            else:
                placeholders_to_labels[placeholder] = [row["template"]]
# print(placeholders_to_labels)
num_labels = []
for i in placeholders_to_labels.keys():
    num_labels.append(len(placeholders_to_labels[i]))

# make a histogram of the number of labels per placeholder
# give a max x-axis value of 20
# have the number of bins be such that the ticks are integers
# make the x-axis ticks be 1, 2, 3, ..., 20
# title it "Number of Unique Labels per Object"
# label the axes "Number of Unique Labels" and "Number of Objects"
# save it to ~/r3m directory
import matplotlib.pyplot as plt
plt.hist(num_labels, bins=20, range=(1,20))
plt.xticks(range(1,21))
plt.title("Number of Unique Labels per Object")
plt.xlabel("Number of Unique Labels")
plt.ylabel("Number of Objects")
plt.savefig(os.path.expanduser("~/r3m/num_labels.png"))

# print top 10 objects with the most labels and their number of labels on the same line
for i in range(10):
    x = max(placeholders_to_labels, key=lambda k: len(placeholders_to_labels[k]))
    y = len(placeholders_to_labels[max(placeholders_to_labels, key=lambda k: len(placeholders_to_labels[k]))])
    del placeholders_to_labels[max(placeholders_to_labels, key=lambda k: len(placeholders_to_labels[k]))]
    print(x, y)
    






