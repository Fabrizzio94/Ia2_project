# import centroid package and trackableObject 
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib

import imutils
import cv2
import numpy as np
import argparse
import utils
import os
import sys
from mrcnn.config import Config
from mrcnn import model as modellib


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

COCO_MODEL_PATH

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
args = vars(ap.parse_args())
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input video file")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output video file")
# ap.add_argument("-m", "--mask-rcnn", required=True,
# 	help="base path to mask-rcnn directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="minimum threshold for pixel-wise mask segmentation")
# args = vars(ap.parse_args())


# load the class label names from disk, one label per line
#CLASS_NAMES = open(args["labels"]).read().strip().split("\n")
CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")



# class InferenceConfig(Config):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"

	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)


# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
# model = modellib.MaskRCNN(mode="inference", config=config,
# 	model_dir=os.getcwd())

#config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
#model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
#        "mrcnn_class_logits", "mrcnn_bbox_fc",
#    "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(COCO_MODEL_PATH, by_name=True)



def random_colors(N):
    np.random.seed(0)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(CLASS_NAMES))
class_dict = {
    name: color for name, color in zip(CLASS_NAMES, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores, num_frames):
    """
        take the image and results and apply the mask, box, and Label
    """
    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    """" - - - - - - - - - -- - - - - - -"""
    # initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
    status = "Waiting"
    rects = []
    """" - - - - - - - - - -- - - - - - -"""

    H = image.shape[0]
    W = image.shape[1]
    n_instances = boxes.shape[0]
    #print(n_instances)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        if label != "car" and label != "person" and label != "motorcycle" and label != "bicycle":
            continue
        
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(image, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    return image


if __name__ == '__main__':
    """
        test everything
    """

    capture = cv2.VideoCapture(0)

    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    
    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        #print(r)

        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores']
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

