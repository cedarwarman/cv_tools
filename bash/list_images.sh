#########################################################
#  cv_tools
#
#  Copyright 2020
#
#  Cedar Warman
#
#  Department of Botany & Plant Pathology
#  Oregon State University
#  Corvallis, OR 97331
#
# This program is not free software; it can be used and modified
# for non-profit only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#########################################################

# Lists only images in a directory of YOLO format annotated images.
ls -1 | grep png | tr -d \*

# To get just the basenames:
# ls -1 | grep png | tr -d .png\*
