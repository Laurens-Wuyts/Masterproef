import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys

save = True
show = True

if "noshow" in sys.argv:
	show = False
	sys.argv.remove("noshow")

if "norec" in sys.argv:
	save = False
	sys.argv.remove("norec")
	if not show:
		print("Not recording or showing won't do anything.")
		exit(1);
else:
	if len(sys.argv) > 2:
		directory = sys.argv[1] + 'Data/' + sys.argv[2] + "/"
	else:
		directory = sys.argv[1] + 'Data/RealsenseTest1'

	if not os.path.exists(directory + "depth/"):
		os.makedirs(directory + "depth/")
	if not os.path.exists(directory + "color/"):
		os.makedirs(directory + "color/")

	dep = [f for f in os.listdir(directory + "depth/") if os.path.isfile(directory + "depth/" + f)]
	idx = len(dep)
	start = idx + 1

print("Rec: %r | Show: %r" % (save, show))

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
	while True:
		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue

		# Convert images to numpy arrays
		depth_image = cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03)
		color_image = np.asanyarray(color_frame.get_data())

		if show:
			depth_scale = cv2.resize(depth_image, (640, 360))
			# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
			depth_colormap = cv2.applyColorMap(depth_scale, cv2.COLORMAP_JET)

			# Stack the three images horizontally
			depth_3ch = cv2.cvtColor(depth_scale, cv2.COLOR_GRAY2BGR)
			images = np.hstack((cv2.resize(color_image, (640, 360)), depth_3ch, depth_colormap))

			# Show images
			cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('RealSense', images)
			cv2.waitKey(1)

		if save:
			idx += 1
			cv2.imwrite(directory + "color/" + str(idx) + ".jpg", color_image)
			cv2.imwrite(directory + "depth/" + str(idx) + ".jpg", depth_image)

finally:
	if save:
		f = open(directory + "shot_frames.txt", "a")
		f.write(str(start) + " - " + str(idx) + "\r\n");
		f.close()
	# Stop streaming
	pipeline.stop()
