from tensorflow.keras.models import load_model
from skimage import io
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path",    required=True, help="path to root of project")
ap.add_argument("-m", "--model",   required=True, help="path to pre-trained network")
ap.add_argument("-d", "--dataset", required=True, help="path to preprocessed dataset")
args = vars(ap.parse_args())

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("[INFO] loading images...")
images = []
depths = []

random.seed()

with h5py.File(args["path"] + "Data/" + args["dataset"], 'r') as f:
	for _ in range(5):
		idx = random.randint(0, f["testX"].shape[0])
		i = f["testX"][idx]
		images.append(i)
		d = f["testX"][idx]
		depths.append(d)

images = np.asarray(images)
depths = np.asarray(depths)
print(images.shape)

print("[INFO] loading model...")
model = load_model(args["path"] + args["model"])
print("[INFO] predicting...")
	
preds = model.predict(images)

preds = 255 * preds
pred_imgs = preds.astype(np.uint8)
print(pred_imgs.shape)

im_preds_color = np.stack((np.squeeze(pred_imgs),)*3, axis=-1)
print(im_preds_color.shape, images.shape)
image = None

for i in range(images.shape[0]):
	temp = np.hstack((images[i], depths[i], im_preds_color[i]))
	if image is None:
		image = temp.copy()
	else:
		image = np.vstack((image, temp))

cv2.imwrite(args["path"] + "Data/predictions.jpg", image)