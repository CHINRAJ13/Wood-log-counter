
Wood Log Counting - v8 wood-log-dataset v2
==============================

This dataset was exported via roboflow.com on October 3, 2023 at 11:14 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 380 images.
Wood-logs are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -30° to +30° horizontally and -5° to +5° vertically
* Random brigthness adjustment of between -36 and +36 percent
* Salt and pepper noise was applied to 7 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random exposure adjustment of between -50 and +50 percent
* Random Gaussian blur of between 0 and 9 pixels


