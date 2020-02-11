#!/usr/bin/env python3

"""
This version will impliment overlap in image subdivisions and non-maximum
suppression. Eventually it will replace the other version, but for now there
are two in case I break this one.

tensorflow_predict.py
Adapted from:
https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb#scrollTo=mz1gX19GlVW7

Usage (note: I've been running this on a Nvidia GPU with Tensorflow 1.12-GPU 
in a conda virtual environment):

bash; conda activate tf1.12-gpu; python tensorflow_predict.py 
    -c <checkpoint_path> 
    -l <label_path> 
    -d <test_image_directory> 
    -o <output_directory>
    -s <min_score_threshold>
    -n <image_split_number>
"""

import os
import glob

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import csv
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Setting up arguments
parser = argparse.ArgumentParser(description='Splits labels')
parser.add_argument('-c', 
                    '--checkpoint',
                    type=str,
                    help=('Path to frozen detection graph (checkpoint)'))
parser.add_argument('-l',
                    '--labels',
                    type=str,
                    default='/home/bpp/warmanma/warman_nfs0/computer_vision/tensorflow/transfer_learning_4/data/label_map.pbtxt',
                    help=('Path to class label map'))
parser.add_argument('-d',
                    '--test_image_dir',
                    type=str,
                    help=('Path to test image directory'))
parser.add_argument('-o',
                    '--output_path',
                    type=str,
                    help=('Path to output directory'))
parser.add_argument('-s',
                    '--min_score_threshold',
                    type=float,
                    default=0.05,
                    help=('Minimum score threshold for plotting bounding boxes'))
parser.add_argument('-n',
                    '--image_split_num',
                    type=int,
                    default=1,
                    help=('Number of image subdivisions to run the object detection on.'))
parser.add_argument('-w',
                    '--overlap_width',
                    type=int,
                    default=100,
                    help=('Pixel overlap width for image subdivisions.'))
args = parser.parse_args()


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = args.checkpoint

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = args.labels

# If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
PATH_TO_TEST_IMAGES_DIR = args.test_image_dir

assert os.path.isfile(PATH_TO_CKPT)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)

# Importing the frozen inference graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loading the images
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# A really complicated fuction to get the split sections, with overlap
def get_splits(image_width, split_number, overlap):

    image_splits = []
    total_image_width = image_width
    overlap_width = overlap

    if args.image_split_num == 1:
        image_splits.append([0, total_image_width]) 
        #print("Lower image split number is: ")
        #print(image_splits[0][0])
        #print("Upper image split number is: ")
        #print(image_splits[0][1])

    # This will be the most used case, as of now, with a split of 3 sub-images.
    # In this case, since a lot of the ear images have significant space on the
    # left and right, I want the center sub-image to not be too big. To avoid
    # this, I'll do the overlaps from the left and right images and leave the
    # center image unchanged.` 
    elif args.image_split_num == 3:
        # Here's the split width if there's no overlap (note: probably will
        # need to do something about rounding errors here with certain image
        # widths).
        no_overlap_width = int(total_image_width / split_number)
        
        # Left split. The left side of the left split will always be zero.
        left_split = []
        left_split.append(0)

        # The other side of the left split will be the width plus the overlap
        left_split.append(no_overlap_width + overlap_width)
        image_splits.append(left_split)

        # The middle has no overlap in this case
        middle_split = []
        middle_split.append(no_overlap_width)
        middle_split.append(no_overlap_width * 2)
        image_splits.append(middle_split)

        # The right split is the opposite of the left split
        right_split = []
        right_split.append((2 * no_overlap_width) - overlap_width)
        right_split.append(total_image_width)
        image_splits.append(right_split)

        # Test prints
        #print("Left split is: " + str(image_splits[0][0]) + ", " + str(image_splits[0][1]))
        #print("Middle split is: " + str(image_splits[1][0]) + ", " + str(image_splits[1][1]))
        #print("Right split is: " + str(image_splits[2][0]) + ", " + str(image_splits[2][1]))

    else:
        # If the split is not 1 or 3, this more general overlap setup happens,
        # with overlaps on all boundaries.
        no_overlap_width = int(total_image_width / split_number)

        # Left split
        left_split = []
        left_split.append(0)
        left_split.append(no_overlap_width + overlap_width)
        image_splits.append(left_split)

        # Middle splits (the minus 2 is because the left and right sides are
        # handled separately)
        for split_position in range(1, (split_number - 1)): 
            middle_split = []
            left_middle_split = (no_overlap_width * split_position) - overlap_width
            right_middle_split = (no_overlap_width * (split_position + 1)) + overlap_width
            middle_split.append(left_middle_split)
            middle_split.append(right_middle_split)
            image_splits.append(middle_split)

        # Right split
        right_split = []
        right_split.append((no_overlap_width * (split_number - 1)) - overlap_width)
        right_split.append(total_image_width)
        image_splits.append(right_split)

        # Test prints (this print only works for split = 4)
        #print("Left split is: " + str(image_splits[0][0]) + ", " + str(image_splits[0][1]))
        #print("Middle split 1 is: " + str(image_splits[1][0]) + ", " + str(image_splits[1][1]))
        #print("Middle split 1 is: " + str(image_splits[2][0]) + ", " + str(image_splits[2][1]))
        #print("Right split is: " + str(image_splits[3][0]) + ", " + str(image_splits[3][1]))
    return(image_splits)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates 
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# This function is for getting out the total number of fluorescent and
# nonfluorescent boxes detected from an output_dict. 
def get_object_counts(output_dict, min_score):
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']

    total_fluorescent = 0
    total_nonfluorescent = 0

    for i in range(len(detection_scores)):
        detect_class = detection_classes[i]
        detect_score = detection_scores[i]
        if detect_score > min_score:
            if detect_class == 1:
                total_fluorescent = total_fluorescent + 1
            if detect_class == 2:
                total_nonfluorescent = total_nonfluorescent + 1

    output_list = [total_fluorescent, total_nonfluorescent]
    return(output_list)

# This function fixes the relative coordinates when splitting an image into
# multiple subimages
def fix_relative_coord(output_dict, image_split_num, image_position):
    output_dict_adj = output_dict

    # First we get a constant adjustment for the "image position". For example,
    # if it's the first image in a series of split images (image 0), then the
    # adjustment would be zero. If it's the second image, the adjustment would
    # be 0.5.
    position_adjustment = image_position * (1 / image_split_num)

    # Now we adjust the x coordinates of the 'detection_boxes' ndarray, We
    # don't need to adjust the y coordinates because we only split on the x. If
    # later I add splitting on y, then the y coordinates need to be adjusted.
    adjusted_boxes = output_dict['detection_boxes']
    adjusted_boxes[:,[1,3]] *= (1 / image_split_num)

    # Adding the adjustment for which split image it is (the first image
    # doesn't need adjustment, hence the if statement).
    if image_position > 0:
        adjusted_boxes[:,[1,3]] += position_adjustment
        

    # Now adding back in the adjusted boxes to the original ndarray
    output_dict_adj['detection_boxes'] = adjusted_boxes

    return(output_dict_adj)

# Setting some stuff up for the totals
image_names = list()
fluorescent_totals = list()
nonfluorescent_totals = list()

# Main basically
for image_path in TEST_IMAGE_PATHS:

    # Sets the image position counter for the relative coordinate fix
    image_position_counter = 0

    image = Image.open(image_path)
    image_name_string = str(os.path.splitext(os.path.basename(image_path))[0])
    print('\nprocessing ' + image_name_string + '\n')
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Here we set up the image splits, based on the users desired number of
    # sub-images, including the overlap. The splits will be a list of sets of
    # two numbers, the lower and upper bounds of the splits.
    splits = get_splits(image_np.shape[1], args.image_split_num, args.overlap_width)
    print(splits)
        
    # Here's where the actual splitting happens
    split_image_np = np.array_split(image_np, args.image_split_num, axis=1)
    
    # Running the inference for the first split, or in the case of a split number
    # of 1, the only split.
    output_dict = run_inference_for_single_image(split_image_np[0], detection_graph)

    # Fixing the relative coordinate with split image problem
    if args.image_split_num > 1:
        output_dict = fix_relative_coord(
            output_dict, 
            args.image_split_num, 
            image_position_counter)
        image_position_counter = image_position_counter + 1

    # Inference for the first split. If there's only one, this is the only one
    # that runs.
    if args.image_split_num > 1:
        # Goes through the image sub-arrays, skipping the first one since we
        # already did that one and the new data will be appended to it.
        for image_split in split_image_np[1:]:
            print("\nProcessing split image. Image position counter:")
            print(str(image_position_counter) + '\n')

            # Running the inference
            split_output_dict = run_inference_for_single_image(image_split, detection_graph)

            # Correcting the relative coordinates
            split_output_dict = fix_relative_coord(
                split_output_dict, 
                args.image_split_num, 
                image_position_counter)

            # Adding the new data to the output dict
            output_dict['detection_boxes'] = np.concatenate((
                output_dict['detection_boxes'], 
                split_output_dict['detection_boxes']))
            output_dict['detection_classes'] = np.concatenate((
                output_dict['detection_classes'], 
                split_output_dict['detection_classes']))
            output_dict['detection_scores'] = np.concatenate((
                output_dict['detection_scores'], 
                split_output_dict['detection_scores']))

            #print(output_dict['detection_boxes'])
            #print(output_dict['detection_classes'])
            #print(output_dict['detection_scores'])

            image_position_counter = image_position_counter + 1

    seed_counts = get_object_counts(output_dict, args.min_score_threshold)

    # Adding in a bit here to count the total number of detections
    #seed_counts = get_object_counts(output_dict, args.min_score_threshold)
    # Adding the numbers to the output lists
    image_names.append(image_name_string)
    fluorescent_totals.append(seed_counts[0])
    nonfluorescent_totals.append(seed_counts[1])

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        max_boxes_to_draw=10000,
        min_score_thresh=args.min_score_threshold)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imsave(args.output_path + '/' + image_name_string + "_plot" + ".jpg", image_np)

# Printing the lists to a file
with open(args.output_path + '/' + 'output.tsv', 'w') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerows(zip(image_names, fluorescent_totals, nonfluorescent_totals))

