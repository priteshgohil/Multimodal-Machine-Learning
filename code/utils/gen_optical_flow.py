#!/usr/bin/env python
import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import cv2
import os
import glob
import numpy as np
import scipy.io as sio
import time

def cvReadGrayImg(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def saveOptFlowToImage(flow, basename, merge):
    if merge:
        # save x, y flows to r and g channels, since opencv reverses the colors
        cv2.imwrite(basename+'.png', flow[:,:,::-1])
    else:
        cv2.imwrite(basename+'_x.png', flow[...,0])
        cv2.imwrite(basename+'_y.png', flow[...,1])

def generate_optical_flow(basepath, fullpath, basename, merge):
    flow_folder = os.path.join(basepath, 'flow', basename)
    cap = cv2.VideoCapture(fullpath)
    ret, img2 = cap.read()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    frame_num = 0
    while True:
        img1 = img2
        ret, img2 = cap.read()
        if not ret:
            break
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = img1.shape
        fxy = norm_width / w
        # normalize image size
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(img1, None, fx=fxy, fy=fxy),
            cv2.resize(img2, None, fx=fxy, fy=fxy), None,
            0.5, 3, 15, 3, 7, 1.5, 0)
        # map optical flow back
        flow = flow / fxy
        # normalization
        flow = np.round((flow + bound) / (2. * bound) * 255.)
        flow[flow < 0] = 0
        flow[flow > 255] = 255
        flow = cv2.resize(flow, (w, h))

        # Fill third channel with zeros
        flow = np.concatenate((flow, np.zeros((h,w,1))), axis=2)

        # save
        if not os.path.isdir(flow_folder):
            os.makedirs(flow_folder)
        flowbasename = "frame_" + "{:04d}".format(frame_num)
        saveOptFlowToImage(flow, os.path.join(flow_folder, flowbasename), merge)
        frame_num += 1

        if args.visual_debug:
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = np.zeros_like(cv2.imread(img_path))
            hsv[...,1] = 255
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('optical flow',bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        if args.visual_debug:
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = np.zeros_like(img2)
            hsv[...,1] = 255
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('optical flow',bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    # duplicate last frame
    flowbasename = "frame_" + "{:04d}".format(frame_num)
    saveOptFlowToImage(flow, os.path.join(flow_folder, flowbasename), merge)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir')
    parser.add_argument('--bound', type=float, required=False, default=15,
                        help='Optical flow bounding. [-bound, bound] will be mapped to [0, 255].')
    parser.add_argument('--merge', dest='merge', action='store_true',
                        help='Merge optical flow in x and y axes into RGB images rather than saving each to a grayscale image.')
    parser.add_argument('--debug', dest='visual_debug', action='store_true',
                        help='Visual debugging.')
    parser.set_defaults(merge=False, visual_debug=False)
    args = parser.parse_args()

    norm_width = 500.
    bound = args.bound

    obj_folders = glob.glob(os.path.join(args.vid_dir, '*'))
    for obj in obj_folders:
        trial_folders = glob.glob(os.path.join(obj, '*'))
        print ("trial_folders=", trial_folders)
        for trial in trial_folders:
            camera_pos_folders = glob.glob(os.path.join(trial, '*'))
            print("camera_pos_folders=",camera_pos_folders)
            for camera_pos in camera_pos_folders:
                exp_folders = glob.glob(os.path.join(camera_pos, '*'))
                print("exp_folders=", exp_folders)
                for exp in exp_folders:
                    videos = glob.glob(os.path.join(exp, '*rgb.mp4'))
                    print("videos = ",videos)
                    for v in videos:
                        print("calculating optical flow for: "+ v)
                        basename = os.path.splitext(os.path.basename(v))[0]
                        generate_optical_flow(exp, v, basename, args.merge)
