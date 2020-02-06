#!/usr/bin/env python3.7

"""
crop_annotations_and_images.py
Cedar Warman

This script takes in images and annotations in PASCAL VOC format and crops them
into smaller images and assocaited annotations. This is to be used as training
data when memory is limiting, which is often the case for maize ear annotated
images with 300-600 bounding boxes per image.

Usage:
crop_annotations_and_images.py 
    -i <input_dir> 
    -o <output_dir>
    -n <image_split_number>

"""

import os
import io
from copy import deepcopy
from pathlib import Path
import numpy as np
import argparse
from lxml import etree

# Setting up arguments
parser = argparse.ArgumentParser(description='Crop annotations and imagmes')
parser.add_argument('-i',
                    '--input_dir',
                    type=str,
                    help=('Path to input directory'))
parser.add_argument('-o',
                    '--output_dir',
                    type=str,
                    help=('Path to output directory'))
parser.add_argument('-n',
                    '--image_split_num',
                    type=int,
                    default=3,
                    help=('Number of subdivisions to crop into.'))
args = parser.parse_args()


"""
=================
Split_annotations
=================
"""

def split_annotations (xml_path, split_number):
    # Importing the xml
    tree = etree.parse(xml_path) 

    # Grabbing the width of the image and making a variable for the current min
    # and max values of the bins.
    image_width = int(tree.xpath('//width')[0].text)
    bin_width = image_width // split_number 
    current_bin_min = 0
    current_bin_max = bin_width

    # Setting up empty lists to contain the split xml files
    tree_list = []
    updated_tree_list = []

    # Copying the tree to the list (each list entry will be the annotations or
    # one sub-image). I need to use deepcopy because all the copies point to
    # the same data, so if any single copy is edited (aka an element is
    # deleted) then all the copies are edited. 
    for x in range(0, split_number):
        tree_to_add = deepcopy(tree)
        tree_list.append(tree_to_add)
 
    # Going through the split trees one by one.
    # TODO Make a counter and some "acceptable" ymin and ymax values the the
    # bounding boxes have to fall into or else they will get deleted, or
    # changed in size if they're over the line. The values will then get
    # something added to them when it does the next tree.

    counter = 0

    for split_tree in tree_list:
        print("\n\n\nProcessing tree\n")

        print("Current bin min: " + str(current_bin_min))
        print("Current bin max: " + str(current_bin_max))

        # Setting up the bin



        # Grabbing the objects from the xml
        object_list = split_tree.xpath('//object')
    
        # Looks at each "object" (aka bounding box, which has the format:
        #   <object>
        #       <name>nonfluorescent</name>
        #       <pose>Unspecified</pose>
        #       <truncated>1</truncated>
        #       <difficult>0</difficult>
        #       <bndbox>
        #           <xmin>208</xmin>
        #           <ymin>693</ymin>
        #           <xmax>265</xmax>
        #           <ymax>746</ymax>
        #       </bndbox>
        #   </object> 
        
        #no_delete_counter = 0
        #delete_counter = 0
        for entry in range(0, len(object_list)):
            current_object = object_list[entry]
            #print(current_object)
            
            xmax = current_object.xpath('bndbox/xmax')[0]
            xmin = current_object.xpath('bndbox/xmin')[0]

            #print(xmax.text)
            #print(xmin.text)
            #print('\n')
            #print("no delete counter = " + str(no_delete_counter))
            #no_delete_counter += 1
            

            if (int(xmin.text) <= current_bin_min or int(xmax.text) >= current_bin_max):
                #print("removing bounding box")
                
                # Removing the entire object from the original tree 
                current_object.getparent().remove(current_object)
        
                counter += 1
                

        current_bin_min = current_bin_min + bin_width
        current_bin_max = current_bin_max + bin_width

        print("Total deleted boxes: " + str(counter))
        counter = 0

    return(tree_list)
        

            


"""
====
Main
====
"""

def main():
    # Setting up the paths
    input_dir = Path(args.input_dir)
    annotations_dir = input_dir / "annotations"

    # Going through each annotation file
    for xml_file in os.listdir(annotations_dir):
        xml_path = str(annotations_dir / xml_file)
        output_tree = split_annotations(xml_path, 3) 
        
        # Printing out the subtrees
        tree_number_print = 1
        for output in output_tree:
            tree_name_output_string = (xml_file[:-4] + "_s" +
                str(tree_number_print) + ".xml")

            xml_string = etree.tostring(output, pretty_print=True)

            with open(tree_name_output_string, 'wb') as f:
                f.write(xml_string)

            tree_number_print += 1













if __name__ == "__main__":
    main()
