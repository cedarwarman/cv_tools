# cv_tools
A collection of short scripts for my maize seed computer vision project.

## Contents
### bash
* list_images.sh
This script lists images in a directory of images & their associated YOLO format annotation files.

### python
* test_tfrecord.py
This script checks a TFRecord file to see what's inside. It prints a huge amount, because I think it's printing all the data that makes up the images, as well as the list of labels and bounding boxes. Work in progress; may be updated in the future to make plots of the images with bounding boxes, and/or list the images the TFRecord contains.

* tidy_yolo_output.py
Takes multi-image yolo output and converts it into a useful format (tab delimited text file of fluorescent and nonfluorescent predicted bounding box coordinates).

* xml_to_yolo.py 
Converts xml output files from the ImageJ Cell Counter plugin to YOLO format annotations.

* split_labels.py
Converts a csv file of image annotations into a training and validation set.

* tensorflow_predict.py
Predicts bounding boxes, total class counts from frozen inference graph. Based on https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb#scrollTo=mz1gX19GlVW7 
