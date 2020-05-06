"""
Mask R-CNN
Train on the Wildfire dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

"""

import os
import sys
import json
import datetime
import time
import numpy as np
import skimage.draw
import imgaug as ia # https://github.com/aleju/imgaug (pip3 install imgaug)
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file

# Weights for ResNet101 otained from https://github.com/matterport/Mask_RCNN/releases
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Weights for ResNet50 obtained from https://github.com/fchollet/deep-learning-models/releases/tag/v0.2
IMAGENET_WEIGHTS_PATH = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class WildfireConfig(Config):
    """Configuration for training on the wildfire dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "wildfire"

    # Google Colab comes with 12GB GPU, we can fit two 1024x1024 default images.
    # Reducing image dimensions to 512x512, we can fit four 512x512 images.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + wildfire

    # Training steps per epoch increased from 100 to 150 
	# so that it is equal to amount of training images.
    STEPS_PER_EPOCH = 150
    
    # Number of validation steps per epoch
    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class WildfireDataset(utils.Dataset):

    def load_wildfires(self, dataset_dir, subset):
        """Load a subset of the wildfire dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Only "wildfire" class is added.
        self.add_class("wildfire", 1, "wildfire")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Annotation loader from coco.py
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "wildfire",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a wildfire dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "wildfire":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "wildfire":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training and Evaluation Functions
############################################################

# Lines 181-192 written by Waleed Abdulla.
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = WildfireDataset()
    dataset_train.load_wildfires(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WildfireDataset()
    dataset_val.load_wildfires(args.dataset, "val")
    dataset_val.prepare()

	# Image Augmentation using imgaug library.
    import imgaug as ia
    from imgaug import augmenters as iaa
    
	# Image Augmentation using imgaug library.
	# Applies a random augmentation sequence 50% of the time.
    augmentations = iaa.Sometimes(0.50, iaa.Sequential([
        iaa.Fliplr(0.50), # Flips images left/right 50% of the time.
        iaa.Crop(percent=(0, 0.2)), # Small crop/zoom on the images.
        iaa.AdditiveGaussianNoise(scale=(0, 40)), # Adds some noise to the images.
        iaa.AddToHueAndSaturation((0, 5))], random_order=True)) # Adds some hue to the images so that wildfire colours slightly vary.

    # Lines 207-214 written by Waleed Abdulla.
    # Training network heads only! More data will allow training of full network in future.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=45,
                layers='heads',
                augmentation=augmentations)

def evaluate_boxes(model):
    """Evaluate the bounding boxes produced by the model."""
    # Prepare the validation dataset for evaluation of the model.
    dataset_val = WildfireDataset()
    dataset_val.load_wildfires(args.dataset, "val")
    dataset_val.prepare()
    all_APs = []
    thresholds = np.linspace(0.5,0.95,10)
	# Iterates across all IoU from 0.5 to 0.95 in steps of 0.05.
    for iou in thresholds:
        APs = []
        t_pred = 0
        t_start = time.time()
        for image_id in dataset_val.image_ids:
            # Loads an image and its ground truth.
            image, image_meta, gt_class_id, gt_bbox, gt_mask =
                modellib.load_image_gt(dataset_val, config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            # Returns detections and time to predict.
            t = time.time()
            results = model.detect([image], verbose=0)
            r = results[0]
            t_pred += (time.time() - t)
            print("Image : {}".format(str(int(image_id)+1)))
            print("Time to predict : {} secs".format(time.time() - t))
            # Compute Average Precision of the image for that IoU threshold.
            AP, precisions, recalls, overlaps =
                utils.compute_box_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], 
                                 r['masks'], iou_threshold = iou)
            APs.append(AP)
            print("AP @ {} IoU = {}\n".format((iou), np.mean(APs)))
            
        all_APs.append(np.mean(APs))
        print("\tTime to predict all images: {} secs, Average of {} secs per image\n".format(
        t_pred, t_pred / len(dataset_val.image_ids)))
        print("\tmAP @ {} IoU = {}\n".format((iou), np.mean(APs)))
      
	# Prints the total to predict 50 images.
    print("\tTotal time: {}".format(time.time() - t_start))
    all_APs.append(np.mean(all_APs))
	# Store the results as a dictionary.
    iou_keys = ["AP @ 0.5", "AP @ 0.55", "AP @ 0.60", "AP @ 0.65", "AP @ 0.7", "AP @ 0.75", 
                "AP @ 0.8", "AP @ 0.85", "AP @ 0.9", "AP @ 0.95", "mAP"]
    dict_APs = dict(zip(iou_keys, all_APs))
    print("\tBox APs across all IoUs: {}".format(dict_APs))
    
def evaluate_masks(model):
    """Evaluate the segmentation masks produced by the model."""
    # Prepare the validation dataset for evaluation of the model.
    dataset_val = WildfireDataset()
    dataset_val.load_wildfires(args.dataset, "val")
    dataset_val.prepare()
    all_APs = []
    thresholds = np.linspace(0.5,0.95,10)
    # Iterates across all IoU from 0.5 to 0.95 in steps of 0.05.
	for iou in thresholds:
        APs = []
        t_pred = 0
        t_start = time.time()
        for image_id in dataset_val.image_ids:
            # Loads an image and its ground truth.
            image, image_meta, gt_class_id, gt_bbox, gt_mask =
                modellib.load_image_gt(dataset_val, config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            # Returns detections and time to predict.
            t = time.time()
            results = model.detect([image], verbose=0)
            r = results[0]
            t_pred += (time.time() - t)
            print("Image : {}".format(str(int(image_id)+1)))
            print("Time to predict : {} secs".format(time.time() - t))
            # Compute Average Precision of the image for that IoU threshold.
            AP, precisions, recalls, overlaps =
                utils.compute_mask_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], 
                                 r['masks'], iou_threshold = iou)
            APs.append(AP)
            print("AP @ {} IoU = {}\n".format((iou), np.mean(APs)))
            
        all_APs.append(np.mean(APs))
        print("\tTime to predict all images: {} secs, Average of {} secs per image\n".format(
        t_pred, t_pred / len(dataset_val.image_ids)))
        print("\tmAP @ {} IoU = {}\n".format((iou), np.mean(APs)))
    
	# Prints the total to predict 50 images.
    print("\tTotal time: {} seconds".format(time.time() - t_start))
    all_APs.append(np.mean(all_APs))
	# Store the results as a dictionary.
    iou_keys = ["AP @ 0.5", "AP @ 0.55", "AP @ 0.60", "AP @ 0.65", "AP @ 0.7", "AP @ 0.75", 
                "AP @ 0.8", "AP @ 0.85", "AP @ 0.9", "AP @ 0.95", "mAP"]
    dict_APs = dict(zip(iou_keys, all_APs))
    print("\tMask APs across all IoUs: {}".format(dict_APs))
    
############################################################
#  Training/Evaluation
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect wildfires.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/wildfire/dataset/",
                        help='Directory of the wildfire dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WildfireConfig()
    else:
        class InferenceConfig(WildfireConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
        model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
        model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        #if not os.path.exists(weights_path):
        #    utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        #weights_path = model.get_imagenet_weights()
        weights_path = IMAGENET_WEIGHTS_PATH
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or Evaluate Boxes/Masks
    if args.command == "train":
        # Training dataset.
        train(model)
    elif args.command == "evaluate_boxes":
        # Validation dataset
        evaluate_boxes(model)
    elif args.command == "evaluate_masks":
        # Validation dataset
        evaluate_masks(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
