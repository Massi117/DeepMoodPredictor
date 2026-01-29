import csv
import random

with open('masks/MVP_rois/desikanKillianyNodeNames.txt', 'r', encoding='utf-8') as file:
    names = file.read()

with open('masks/MVP_rois/desikanKillianyNodeIndex.1D', 'r', encoding='utf-8') as file:
    index = file.read()

data = zip(names.splitlines(), index.splitlines())

for row in data:

    # Open the file in append mode ('a')
    with open('masks/MVP_rois/atlas_index.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the new list of values as a single row
        writer.writerow(row)