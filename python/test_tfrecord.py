#!/usr/bin/env python3

"""
test_tfrecord
Cedar Warman

Checks a tfrecord file to see what's inside.

Caution: prints a huge amount (maybe it's printing all the data contained in the images?)
"""

import tensorflow as tf
import argparse


# Setting up argparse
parser = argparse.ArgumentParser(description='Check tfrecord file')
parser.add_argument('-i', 
                    '--input',
                    metavar='',
                    help=('Input tfrecord file'),
                    type=str)
args = parser.parse_args()

for example in tf.python_io.tf_record_iterator(args.input):
    print(tf.train.Example.FromString(example))
