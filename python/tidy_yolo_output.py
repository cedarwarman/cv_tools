#!/usr/bin/env python3

"""
tidy_yolo_output
Cedar Warman

Here we take multi-image yolo output and sum fluorescent and nonfluorescent 
classes into a format that we can use.

"""

import sys
import io
import re
import csv
import argparse

# Setting up argparse
parser = argparse.ArgumentParser(description='Process multi-image yolo output')
parser.add_argument('-i', 
                    '--input',
                    metavar='',
                    help=('Input text file (yolo multi-image output)'),
                    type=str)
args = parser.parse_args()

def process_yolo_output(input_file):
    fandle = io.open(input_file, 'r') # opening up the file

    # First image flag (keeps the loop from adding counts to the fluor/nonfluor
    # lists if it's the first time through
    not_first_image = 0

    # Some lists to hold the names and counts (with column titles)
    image_names = list()
    image_names.append('image_name')
    fluor_count = list()
    fluor_count.append('fluor_sum')
    nonfluor_count = list()
    nonfluor_count.append('nonfluor_sum')

    # Temp fluor/nonfluor counters (will have the totals for each image, then
    # cleared)
    current_fluor = 0
    current_nonfluor = 0

    for line in fandle:
        line_stripped = line.strip()
        # Checks to see if it's a new image yet
        if re.search(r'^Enter Image Path: /', line_stripped):
            # Adding the previous sums to the relevant lists
            if not_first_image:
                fluor_count.append(current_fluor)
                nonfluor_count.append(current_nonfluor)

            # Now it is the first image, so
            not_first_image = 1

            # Clearing the fluor/nonfluor sum lists
            current_fluor = 0
            current_nonfluor = 0

            # Pulling out the image name
            image_name_string = str(re.search(r'(?<=./)X[\d]+.*(?=\.png)',
                                              line_stripped).group(0))

            image_names.append(image_name_string)

        if re.search(r'^fluorescent', line_stripped):
            current_fluor = current_fluor + 1
            
        if re.search(r'^nonfluorescent', line_stripped):
            current_nonfluor = current_nonfluor + 1

    # This should grab the last sums and add them to the list
    fluor_count.append(current_fluor)
    nonfluor_count.append(current_nonfluor)

    fandle.close()

    # Writing out the final lists
    with open('output.tsv', 'w') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerows(zip(image_names, fluor_count, nonfluor_count))


def main():
    process_yolo_output(args.input)

if __name__ == '__main__':
    main()

