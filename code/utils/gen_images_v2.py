#!/usr/bin/env python
import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
import glob
import numpy as np


def generate_images(basepath, videofile, basename, fps, w, h):
	image_folder = os.path.join(basepath, 'images', basename)
	# "ffmpeg -i front_rgb.mp4 -s 320x240 -vf fps=18 frame_%04d.jpg"
	print("creating images")
	if not os.path.isdir(image_folder):
		os.makedirs(image_folder)
	os.system("ffmpeg -i {} -s {}x{} -vf fps={} {}/frame_%04d.jpg".format(videofile, w, h, fps, image_folder))	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--vid_dir')
	parser.add_argument('--fps', type=int, required=True, default=1, help='frame per second to capture in video')
	parser.add_argument('--H', type=int, required=True, help='frame per second to capture in video')
	parser.add_argument('--W', type=int, required=True, help='frame per second to capture in video')
	args = parser.parse_args()

	obj_folders = glob.glob(os.path.join(args.vid_dir, '*'))
	for obj in obj_folders:
		trial_folders = glob.glob(os.path.join(obj, '*'))
		for trial in trial_folders:
			camera_pos_folders = glob.glob(os.path.join(trial, '*'))
			for camera_pos in camera_pos_folders:
				exp_folders = glob.glob(os.path.join(camera_pos, '*'))
				for exp in exp_folders:
					videos = glob.glob(os.path.join(exp, 'front_rgb.mp4'))
					for v in videos:
						print("calculating optical flow for: "+ v)
						basename = os.path.splitext(os.path.basename(v))[0]
						generate_images(exp, v, basename, args.fps, args.W, args.H)
