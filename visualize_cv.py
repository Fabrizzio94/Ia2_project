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
#config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
# #model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
# #        "mrcnn_class_logits", "mrcnn_bbox_fc",
# #    "mrcnn_bbox", "mrcnn_mask"])
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
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}


"""" - - - - - - - - - -- - - - - - -"""

def display_instances(image, boxes, masks, ids, names, scores, nFrames, totalDown, totalUp, trackers):
    """
        take the image and results and apply the mask, box, and Label
    """
    
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
    
    if nFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []
        for i in range(n_instances):
            
            
            if not np.any(boxes[i]):
                continue

            y1, x1, y2, x2 = boxes[i]
            label = names[ids[i]]
            # if label != "car" and label != "person" and label != "motorcycle" and label != "bicycle":
            #     continue
            if label != "car":
                continue
            color = class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]

            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # add the bounding box coordinates to the rectangles list
            # rects.append((x1, y1, x2, y2))
            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x1, y1, x2, y2)
            tracker.start_track(image, rect)
            # add the tracker to our list of trackers so we can
			# utilize it during skip frames
            trackers.append(tracker)

            # draw the score on the left corner of rectangle
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
            status = "Tracking"
            # update the tracker and grab the updated position
            tracker.update(image)
            pos = tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(image, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    #print(len(rects))
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
		# object ID
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
        else:
            # the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

			# check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

		# store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 31, 26), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (4, 31, 26), -1)
    # construct a tuple of information we will be displaying on the
	# frame
    info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image, totalDown, totalUp, trackers


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

