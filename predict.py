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

random.seed()

with h5py.File(args["path"] + "Data/" + args["dataset"], 'r') as f:
	for _ in range(5):
		i = f["testX"][random.randint(0, f["testX"].shape[0])]
		images.append(i)

images = np.asarray(images)
print(images.shape)

print("[INFO] loading model...")
model = load_model(args["path"] + args["model"])
print("[INFO] predicting...")
	
preds = model.predict(images)

preds = 255 * preds
pred_imgs = preds.astype(np.uint8)
print(pred_imgs[0].shape)

im_pred_color = cv2.cvtColor(pred_imgs[0], cv2.GRAY2BGR)
cv2.imwrite(args["path"] + "Data/predictions.jpg", np.hstack((images[0], im_pred_color)))