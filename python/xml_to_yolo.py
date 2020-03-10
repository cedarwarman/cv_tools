#!/usr/bin/env python3

#########################################################
#  cv_tools
#
#  Copyright 2020
#
#  Cedar Warman
#
#  Department of Botany & Plant Pathology
#  Oregon State University
#  Corvallis, OR 97331
#
# This program is not free software; it can be used and modified
# for non-profit only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#########################################################

"""
xml_to_yolo
Cedar Warman

This script converts xml output files from the ImageJ Cell Counter plugin to
YOLO format annotation files.
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse

"""
========================
Setting up the arguments
========================
"""

parser = argparse.ArgumentParser(description='Given xml file of seed locations, returns YOLO formatted annotations')
parser.add_argument('-i', '--input_xml_path', type=str, help='Path of input xml file. Should be ImageJ Cell Counter output')
parser.add_argument('-b', '--box_size', type=float, default=70, help='Pixel edge length of box')
args = parser.parse_args()


"""
===================
xml parser function
===================
Takes an xml file and returns a list of category and x/y coordinates for each 
kernel.
"""

def parse_xml (input_xml):
    # Make element tree for object
    tree = ET.parse(input_xml)

    # Getting the root of the tree
    root = tree.getroot()

    # Pulling out the name of the image
    image_name_string = (root[0][0].text)

    # Pulling out the fluorescent and non-fluorescent children
    fluorescent = root[1][1]
    nonfluorescent = root[1][2]

    # Setting up some empty lists to move the coordinates from the xml into
    fluor_x = []
    fluor_y = []
    nonfluor_x = []
    nonfluor_y = []

    # Getting the coordinates of the fluorescent kernels
    for child in fluorescent:
        if child.tag == 'Marker':
            fluor_x.append(child.find('MarkerX').text)
            fluor_y.append(child.find('MarkerY').text)

    # Getting the coordinates of the non-fluorescent kernels
    for child in nonfluorescent:
        if child.tag == 'Marker':
            nonfluor_x.append(child.find('MarkerX').text)
            nonfluor_y.append(child.find('MarkerY').text)

    # Putting together the results for output
    fluor_coord = np.column_stack((fluor_x, fluor_y))
    nonfluor_coord = np.column_stack((nonfluor_x, nonfluor_y))

    return_list = [fluor_coord, nonfluor_coord, image_name_string]
    return(return_list)


"""
===================
Convert xml to YOLO
===================
Takes parsed xml files and builds a list of YOLO formatted annotations, which
can then be printed.
"""

def xml_to_yolo(input_coordinates, input_box_size):
    # Naming/extracting the input variables
    fluor = input_coordinates[0]
    nonfluor = input_coordinates[1]
    image_name = input_coordinates[2]
    box_edge_length = int(input_box_size)

    # Setting up the image dimensions. For my images they are all the same, but
    # be sure to change these lines if you're using images that don't match my
    # dimensions.
    image_width = 1920
    image_height = 746

    # This is for writing out at the end. Each line will be a YOLO annotation.
    output_string_list = []

    # Going through the fluorescent kernels and making the annotations
    for index, coord in enumerate(fluor):
        x = int(coord[0])
        y = int(coord[1])

        # Finding the relative x and y coordinates of the center of the box
        rel_x = x / image_width
        rel_y = y / image_height

        # Finding the relative box size (note: for this script the box size is
        # constant, depending in the -b input)
        rel_box_edge_width = box_edge_length / image_width
        rel_box_edge_height = box_edge_length / image_height

        coord_string = ('0' +
                        ' ' +
                        str(rel_x) +
                        ' ' +
                        str(rel_y) +
                        ' ' +
                        str(rel_box_edge_width) +
                        ' ' +
                        str(rel_box_edge_height))

        output_string_list.append(coord_string)

    # Doing the same thing for the non-fluorescent kernels.
    for index, coord in enumerate(nonfluor):
        x = int(coord[0])
        y = int(coord[1])

        # Finding the relative x and y coordinates of the center of the box
        rel_x = x / image_width
        rel_y = y / image_height

        # Finding the relative box size (note: for this script the box size is
        # constant, depending in the -b input)
        rel_box_edge_width = box_edge_length / image_width
        rel_box_edge_height = box_edge_length / image_height


        coord_string = ('1' +
                        ' ' +
                        str(rel_x) +
                        ' ' +
                        str(rel_y) +
                        ' ' +
                        str(rel_box_edge_width) +
                        ' ' +
                        str(rel_box_edge_height))

        output_string_list.append(coord_string)

    return(output_string_list)


"""
============
List printer
============
Takes xml_to_yolo output and prints the list to a file.
"""

def print_list(command_list, image_name_input):
    # Setting up the output file name
    image_name = image_name_input
    image_name_string = (image_name_input.rstrip('.png') +
                         ".txt")

    file = open(image_name_string, "w+")

    for command in command_list:
        file.write("%s\n" % command)

    file.close()



"""
==========
Running it
==========
"""

def main():
    coordinates = parse_xml(args.input_xml_path)
    command_list = xml_to_yolo(coordinates, args.box_size)
    print_list(command_list, coordinates[2])

if __name__ == '__main__':
    main()

