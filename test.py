import csv
import random

new_row_values = [random.randint(0, 100), random.random()]

# Open the file in append mode ('a')
with open('data/accuracies.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the new list of values as a single row
    writer.writerow(new_row_values)