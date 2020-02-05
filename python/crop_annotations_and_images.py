#!/usr/bin/env python3.7

# crop_annotations_and_images.py

"""
crop_annotations_and_images.py
Cedar Warman

This script takes in images and annotations in PASCAL VOC format and crops them
into smaller images and assocaited annotations. This is to be used as training
data when memory is limiting, which is often the case for maize ear annotated
images with 300-600 bounding boxes per image.

Usage:
crop_annotations_and_images.py -i <input_dir> -n <image_split_number>

"""
