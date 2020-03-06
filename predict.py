from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="path to pre-trained traffic sign recognizer")
ap.add_argument("-i", "--images", required=True,help="path to testing directory containing images")
args = vars(ap.parse_args())

DATA_PATH = "/home/laurens/masterproef/Data/GTSRB"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("[INFO] loading model...")
model = load_model(args["model"])
labelNames = open(os.path.sep.join([DATA_PATH, "signnames.csv"])).read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]
print("[INFO] predicting...")
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:25]

for (i, imagePath) in enumerate(imagePaths):
	image = io.imread(imagePath)
	image = transform.resize(image, (32, 32))
	image = exposure.equalize_adapthist(image, clip_limit=0.1)
	
	image = image.astype("float32") / 255.0
	image = np.expand_dims(image, axis=0)
	preds = model.predict(image)
	j = preds.argmax(axis=1)[0]
	label = labelNames[j]
	
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=128)
	cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	p = os.path.sep.join(["examples/", "{}.png".format(i)])
	cv2.imwrite(p, image)