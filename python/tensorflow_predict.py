#!/usr/bin/env python3

"""
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
                    help=('Path to frozen detection graph (checkpint)'))
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
#print(TEST_IMAGE_PATHS)

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


# Setting some stuff up for the totals
image_names = list()
fluorescent_totals = list()
nonfluorescent_totals = list()

# Main basically
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_name_string = str(os.path.splitext(os.path.basename(image_path))[0])
    print('\nprocessing ' + image_name_string + '\n')
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    split_image_np = np.vsplit(image_np, args.image_split_num)
    
    ### TESTING BLOCK
    #print('\n' + 'image_np_shape' + '\n')
    #print(image_np.shape)
    #print('\n' + 'Split array shape:' + '\n')
    #print(split_image_np[0].shape)
    #print(split_image_np[1].shape)
    #print('\n' + 'Split array total print:' + '\n')
    #print(split_image_np)

    ### END TESTING BLOCK


    output_dict = run_inference_for_single_image(split_image_np[0], detection_graph)

    # For loop will be like, for item in list of sub-arrays, do detection, then
    # append output_dict stuff to the old output_dict

    # Or maybe better, so that it builds the output_dict first, it should do
    # the first division, then check to see if there are more, then do the
    # remaining ones in a for loop if necessary. Then you can append the
    # relevant secitons of the output_dict (eg. detection_boxes, classes,
    # scores) to the output dict.
    print(output_dict['detection_boxes'])
    print(output_dict['detection_classes'])
    print(output_dict['detection_scores'])



    # Inference for the first split. If there's only one, this is the only
    if args.image_split_num > 1:
        # Goes through the image sub-arrays, skipping the first one since we
        # already did that one and the new data will be appended to it.
        for image_split in split_image_np[1:]:
            print("\nProcessing split image")
            split_output_dict = run_inference_for_single_image(image_split, detection_graph)
            # Here will be a series of appends (**concatenations) to the output_dict to add the
            # new data
            output_dict['detection_boxes'] = np.concatenate((
                output_dict['detection_boxes'], 
                split_output_dict['detection_boxes']))
            output_dict['detection_classes'] = np.concatenate((
                output_dict['detection_classes'], 
                split_output_dict['detection_classes']))
            output_dict['detection_scores'] = np.concatenate((
                output_dict['detection_scores'], 
                split_output_dict['detection_scores']))

            print(output_dict['detection_boxes'])
            print(output_dict['detection_classes'])
            print(output_dict['detection_scores'])


        


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
        #instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=6,
        max_boxes_to_draw=10000,
        min_score_thresh=args.min_score_threshold)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imsave(args.output_path + '/' + image_name_string + "_plot" + ".jpg", image_np)

# Testing the totals lists
#print(image_names)
#print(fluorescent_totals)
#print(nonfluorescent_totals)
    
# Printing the lists to a file
with open(args.output_path + '/' + 'output.tsv', 'w') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerows(zip(image_names, fluorescent_totals, nonfluorescent_totals))










