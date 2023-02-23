import json
import csv
import os

# create new csv file with headers txt, len, and path
# Check if file exists
if os.path.isfile('/home/ademi_adeniji/r3m/drawer_manifest_eval.csv'):
    # if so, delete it
    os.remove('/home/ademi_adeniji/r3m/drawer_manifest_eval.csv')

with open('/home/ademi_adeniji/r3m/drawer_manifest_eval.csv', 'w') as csvfile:
    fieldnames = ['id', 'txt', 'len', 'path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# load json file as a list of dictionaries
with open('/shared/ademi_adeniji/something-something/something-something-v2-validation.json') as json_file:
    data = json.load(json_file)
    # iterate through each dictionary in the list
    for entry in data:
        # check if the label field contains the word "drawer"
        if "opening drawer" in entry['label'] or "opening a drawer" in entry['label'] or "closing drawer" in entry['label'] or "closing a drawer" in entry['label']:
            # if so get the id of the entry
            id = entry['id']
            # store the label field in a variable called txt
            txt = entry['label']
            # check if a folder called "id" exists in /shared/ademi_adeniji/something-something-dvd/
            if os.path.isdir('/shared/ademi_adeniji/something-something-dvd/' + id):
                # if so store the path to the folder in a variable
                path = '/shared/ademi_adeniji/something-something-dvd/' + id
                # get the number of files in the folder
                frames = len(os.listdir(path))
                # write id, txt, frames, and path to the csv file as id, txt, len, and path
                with open('drawer_manifest_eval.csv', 'a') as csvfile:
                    fieldnames = ['id', 'txt', 'len', 'path']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'id': id, 'txt': txt, 'len': frames, 'path': path})

# save the csv file
csvfile.close()







                




