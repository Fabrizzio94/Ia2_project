#import import_ipynb

import cv2
import numpy as np
import sys
import os
#import visualize_cv
from visualize_cv import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")
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


config = SimpleConfig()
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
# model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
#         "mrcnn_class_logits", "mrcnn_bbox_fc",
#     "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Initialize video capture from video file
capture = cv2.VideoCapture('videos/UTM.mp4')
# try to determine the total number of frames in the video file
try:
	# prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
    #     else cv2.CAP_PROP_FRAME_COUNT
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(capture.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('output/video_counter_utm_1.avi', codec, 30.0, size)
counter = 0
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        #print(results)
        r = results[0]
        #rects = []
        frame, rects = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'], counter
        )
        output.write(frame)
        cv2.imshow('frame', frame)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()

