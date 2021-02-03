import torch.utils.data as data
import numpy as np
import glob
import os
import sys
import torch

from PIL import Image
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


def video_to_tensor(frames):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(frames.transpose([3,0,1,2]))

def preprocess_image(cv_frame, size):
    """
    Args:   frame - cv image
            size - tuple of (W x H)
    """
    im = cv2.resize(cv_frame, size) #compress image
    im = np.array(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))) #convert to RGB image
    im = im/255 #normalize
    im = (im-0.5)/0.5
    return im

def video_loader(vid_path, frames):
    """
    Args:   vid_path - location of saved video frames
            frames - numpy array of frame number to read
    """
    resize_W = 160
    resize_H = 120
    images = []
    for frame in vid_frames:
        image_name = "frame_{:04n}.jpg".format(frame)
        img = cv2.imread(vid_path + image_name)
        img = preprocess_image(img, (resize_W,resize_H))
        images.append(img)
    return np.array(images)

def split_frames(start,stop,length):
    """
    To match video frames with tactile,
    Input:  start - starting frame of video
            stop - last frame of video
    output: sequence of frames in batch of length.
    for ex: If length = 18, start = 77 and stop = 300 then
            video is divided into 13 clips and frame number is returned.
    """
    vid_sample = []
    tac_sample = []
    frames = np.arange(start,stop)
    groups = len(frames)//length       #we will get these many clips from given video frames,
    extra_frames = len(frames)%length  #need to fillup missing frames,
    for index, i in enumerate(frames[::length]):
        if(i+length<stop):
            vid_sample.append(np.arange(i,i+length))
            tac_sample.append(np.arange(index*length,index*length+length))
        else:  #now append missing frames
            end_frames = np.arange(i,stop)
            missing_frames = length-end_frames.shape[0]
            end_frames = np.append(end_frames, np.full((missing_frames,),stop))
            vid_sample.append(end_frames)
            tac_sample.append(np.arange(index*length,index*length+length))
    if (vid_sample==[]):
        raise ValueError("Not enough frames in a video")
    if (len(vid_sample) != groups and extra_frames and len(vid_sample) != (groups+1)):
        raise ValueError("something went wrong. Expexted {} splits, but have only {}".format(groups,len(sample)))
    return vid_sample,tac_sample

def get_offset(annotation_path, label_path):
    """
    Get the offset in samples between video and tactile data
    Args:   annotation_path - annotation.txt file containinf video frame alligned with label[0]
            label_path - path of label.txt
    """
    vid_fps = 18
    tac_fps = 16.67
    annotation = np.loadtxt(annotation_path)
    label = np.loadtxt(label_path)
    return int(annotation - label[0] * (vid_fps/tac_fps))

def get_label(pickup, drop, vid_frame, length, actual_label):
    """
    Get the lable for given clip of video
    Args:   pickup - label[0]
            drop - label[2]
            vid_frame - numpy array of video frames
            length - length of clip
            actual_lable - label[3]
    """
    label = None
    label_range = np.arange(pickup, drop)
    #set lable to maximum frames follow
    temp_label = np.empty(length)
    temp_label[np.in1d(vid_frame,label_range)]==1.0
    if(np.where(temp_label==1.0)[0].shape[0] > length//2):
        label = actual_label
    else:
        label = 1
    return np.array(label,dtype=float)

def make_train_dataset(data_path,req_frame_length):
    data = []
    for path in data_path:
        front_vid_path, tac_path, pos_path, label_path, annotation_path = path[0], path[2], path[3], path[-1], path[6]

        offset = get_offset(annotation_path, label_path)
        frames = glob.glob(os.path.join(front_vid_path,'*.jpg'))
        video_frames = np.arange(offset, len(frames)+1)
        start,stop = video_frames[0], video_frames[-1]
        if(len(video_frames) >= 430):
            raise ValueError("more image frames = {} than tactile data for video {}".format(video_frames, front_vid_path))
        #collect subsequent frame numbers
        vid_frames, tac_frames = split_frames(start, stop, req_frame_length)
        label = np.loadtxt(label_path)
        for vid_frame, tac_frame in zip(vid_frames, tac_frames):
            pickup, drop = label[0]*18/16.67, label[2]*18/16.67
            sequence_label = get_label(pickup, drop, vid_frame, req_frame_length, label[3])
            data.append((front_vid_path, tac_path, pos_path, label_path, vid_frame, tac_frame, sequence_label))
    return data

def get_location(root, split_file):
    data_path = []
    file_path = np.loadtxt(split_file,dtype=str)
    prefix = root.split(file_path[0][:32])[0]
    file_names = ['images/front_rgb/','images/left_rgb/','tactile.txt','pos.txt','label.txt','flow', 'video_grasp_timestamp.txt']
    for i,file in enumerate(file_path):
        front_video_path = prefix + file + file_names[0]
        left_video_path = prefix + file + file_names[1]
        tactile_path = prefix +"/Visual-Tactile_Dataset/tactile_data/"+file.split(file[:32])[1] + file_names[2]
        pos_path = prefix + file + file_names[3]
        label_path = prefix + file + file_names[4]
        front_flow_path = prefix + file + file_names[5] + '/' + 'front_rgb/'
        left_flow_path = prefix + file + file_names[5] + '/' + 'left_rgb/'
        annotation_path = prefix + "/Visual-Tactile_Dataset/dataset_annotations/"+file.split(file[:32])[1] + file_names[6]
        data_path.append((front_video_path, left_video_path, tactile_path, pos_path, \
                       front_flow_path, left_flow_path, annotation_path, label_path))
    return data_path



class VisualTactile(data.Dataset):
    """
    Dataset to load sequence of framses frm the video
    root = directory of ur datasets
    split_file = your test or train split .txt file
    transforms = Transform if you wants to apply to video
    frames_to_load = 18 default, sequence of frames to load
    """

    def __init__(self, root, split_file, transforms=None, frames_to_load = 18):
        self.root = root
        self.split_file = split_file
        self.transforms = transforms
        self.frames_to_load = frames_to_load
        # get path of each file data you would like to process
        self.data_path = get_location(self.root, self.split_file)
        if len(self.data_path) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        # get path and frames for training data
        if("train" in split_file):
            self.data = make_train_dataset(self.data_path, self.frames_to_load)
#         else:
#             self.data = test_loader(self.data_path)

    def __getitem__(self,index):
        front_vid_path, tac_path, pos_path, label_path, vid_frames, tac_frames, label = self.data[index]
        clip = video_loader(front_vid_path, vid_frames)
        return video_to_tensor(clip), torch.from_numpy(label)

    def __len__(self):
        return 0

    def get_video_frames(self):
        return 0

# obj = VisualTactile("../../../t/Visual-Tactile_Dataset/dataset/", "../master_i3d/trainv2.txt")

path1 = ('../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/right/4/images/front_rgb/', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/right/4/images/left_rgb/', '../../../t/Visual-Tactile_Dataset/tactile_data/Cheez/50_432/right/4/tactile.txt', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/right/4/pos.txt', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/right/4/flow/front_rgb/', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/right/4/flow/left_rgb/', '../../../t/Visual-Tactile_Dataset/dataset_annotations/Cheez/50_432/right/4/video_grasp_timestamp.txt', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/right/4/label.txt')
path2 = ('../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/top/7/images/front_rgb/', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/top/7/images/left_rgb/', '../../../t/Visual-Tactile_Dataset/tactile_data/Cheez/50_432/top/7/tactile.txt', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/top/7/pos.txt', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/top/7/flow/front_rgb/', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/top/7/flow/left_rgb/', '../../../t/Visual-Tactile_Dataset/dataset_annotations/Cheez/50_432/top/7/video_grasp_timestamp.txt', '../../../t/Visual-Tactile_Dataset/dataset/Cheez/50_432/top/7/label.txt')
path = [path1,path2]
data = make_train_dataset(path, 18)
