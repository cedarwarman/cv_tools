# Lists only images in a directory of YOLO format annotated images.
ls -1 | grep png | tr -d \*

# To get just the basenames:
# ls -1 | grep png | tr -d .png\*
