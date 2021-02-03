import torch
import torch.utils.data as data
from PIL import Image
import os

import glob
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import datetime

def clip_loader(path, start_frame_num, length, \
                input_width=160, \
                input_height=120, \
                output_width=112, \
                output_height=112):
    '''
    loads `length`-frame clip starting from frame_num from
    video specified in path
    - if not enough frames, copies last frame multiple times
    - scales frame to input_width x input_height
    - crops to output_width x output_height in center or corners
    - 50% chance of horizontal flip
    - images will be in range -0.5 to 0.5
    '''
    w = output_width
    h = output_height

    ow = input_width
    oh = input_height
    # crop in corners or center
    starts = [(0,0), (oh-h, 0), (0, ow-w), (oh-h, ow-w), (oh/2 - h/2, ow/2 - w/2)]
    startpt = starts[np.random.randint(0, 4)]
    y = int(startpt[0])
    x = int(startpt[1])
    # horizontal flip
    flip = np.random.randint(0, 1)
    clip = []
    frames_to_fetch_r = range(start_frame_num, start_frame_num+length, 1)
    frames_to_fetch = [f for f in frames_to_fetch_r]
    for frame_num in frames_to_fetch:
        img_filename = os.path.join(path, 'frame_{:04d}'.format(frame_num))
        img_filename += '.jpg'
        frame = cv2.imread(img_filename, cv2.IMREAD_COLOR)
        # copy last frame if not enough frames
        if frame is None:
            frame = clip[-1]
            clip.append(frame)
            continue
        # resize to input size
        frame = cv2.resize(frame, (input_width, input_height))
        # randomly flip
        if (flip == 1):
            frame = np.flip(frame, 1)
        # crop
        frame = frame[y:y+h, x:x+w]
        frame = frame.astype(np.float32) / 255.0
        clip.append(frame)
    clip = np.array(clip)
    clip -= 0.5
    clip = np.moveaxis(clip, -1, 0)
    return clip


def sensor_data_loader(data, start_frame_num, length):
    '''
    load sensor data from start_frame_num
    if required length is not available in data, the
    last row of data is copied until required length
    '''
    if start_frame_num + length <= data.shape[0]:
        d = data[start_frame_num:start_frame_num+length]
    else:
        d = data[start_frame_num:]
        repeat_length = start_frame_num + length - data.shape[0]
        dx = np.tile(d[-1], repeat_length).reshape(repeate_length, d.shape[1])
        d = np.vstack((d, dx))
    # remove first column (timestamp)
    d = d[:, 1:]
    return d

def get_data(root_dir, dir_list):
    video_folders = []
    wrench_data = []
    js_pos_data = []
    js_vel_data = []
    js_effort_data = []
    base_vel_data = []
    frame_counts = []
    image_timestamps = []
    labels = []
    frame_labels = []
    for d in dir_list:
        full_path = os.path.join(root_dir, d)
        video_folder = os.path.join(full_path, 'head') #Select your camera here
        video_folders.append(video_folder)
        if ('success' in video_folder):     #Select labels here
            labels.append(1.0)
        else:
            labels.append(0.0)
        frame_count = len(glob.glob(video_folder + '/*.jpg')) #total images in folder
        frame_counts.append(frame_count)
        img_timestamps_file = os.path.join(full_path, 'image_timestamps.npy')
        wrench_file = os.path.join(full_path, 'wrench_resampled.npy')
        js_pos_file = os.path.join(full_path, 'joint_state_positions_resampled.npy')
        js_vel_file = os.path.join(full_path, 'joint_state_velocities_resampled.npy')
        js_effort_file = os.path.join(full_path, 'joint_state_efforts_resampled.npy')
        base_vel_file = os.path.join(full_path, 'base_velocity_resampled.npy')
        frame_label_file = os.path.join(full_path, 'label.txt')

        img_ts = np.load(img_timestamps_file)
        image_timestamps.append(img_ts)

        wrench = np.load(wrench_file)
        wrench_data.append(wrench)

        js_pos = np.load(js_pos_file)
        js_pos_data.append(js_pos)

        js_vel = np.load(js_vel_file)
        js_vel_data.append(js_vel)

        js_effort = np.load(js_effort_file)
        js_effort_data.append(js_effort)

        base_vel = np.load(base_vel_file)
        base_vel_data.append(base_vel)

        frame_label = np.loadtxt(frame_label_file)
        frame_labels.append(frame_label)

    labels = np.array(labels)
    image_timestamps = np.array(image_timestamps)
    wrench_data = np.array(wrench_data)
    data_dict = dict()
    data_dict['video_folders'] = video_folders
    data_dict['labels'] = labels # video level
    data_dict['frame_counts'] = frame_counts
    data_dict['image_timestamps'] = image_timestamps
    data_dict['wrench'] = wrench_data
    data_dict['joint_state_positions'] = js_pos_data
    data_dict['joint_state_velocities'] = js_vel_data
    data_dict['joint_state_efforts'] = js_effort_data
    data_dict['base_velocities'] = base_vel_data
    data_dict['frame_labels'] = frame_labels # frame level
    return data_dict



def is_clip_anomalous(frame_start, length, frame_labels):
    '''
    if more than 50% of frames have anomalous labels, the clip is
    considered anomalous
    '''
    clip_frames = list(range(frame_start, frame_start + length))
    num_anomalous_frames = len(set(clip_frames).intersection(frame_labels))
    if (num_anomalous_frames > 0.5 * length):
        return False
    else:
        return True


class HSRDataset(data.Dataset):
    def __init__(self, root_dir, dir_list, clip_length):
        self.clip_length = clip_length
        self.data = get_data(root_dir, dir_list)
        self.total_clips = 0
        clip_sums = []
        for c in self.data['frame_counts']: #code to count total number of clips
            clips = c / clip_length
            self.total_clips += int(clips)
            clip_sums.append(self.total_clips)
        self.cummulative_clips = np.array(clip_sums) # total cumulatively added clips in each folder

    def __getitem__(self, index):
        #Which folder to read clip is decided from the video_id
        video_id = np.argmax(self.cummulative_clips > index) #return the index of self.cummulative_clips where index is between that range
        if (video_id == 0): #video_id is the first folder then video_seq is index
            clip_id = index
        else:
            clip_id = index - self.cummulative_clips[video_id - 1]
        # print('clip id ', clip_id)
        frame_start = clip_id * self.clip_length
        video_folder = self.data['video_folders'][video_id]
        video_seq = clip_loader(video_folder, frame_start, self.clip_length)
        label = self.data['labels'][video_id]

        item = dict()
        item['video_id'] = video_id
        item['clip'] = torch.from_numpy(video_seq)
        item['label'] = torch.from_numpy(np.array(label,dtype=float))
        item['wrench'] = torch.from_numpy(sensor_data_loader(self.data['wrench'][video_id],
                                                              frame_start, self.clip_length).T)
        item['joint_state_position'] = torch.from_numpy(sensor_data_loader(self.data['joint_state_positions'][video_id],
                                                          frame_start,
                                                          self.clip_length))
        item['joint_state_velocity'] = torch.from_numpy(sensor_data_loader(self.data['joint_state_velocities'][video_id],
                                                          frame_start,
                                                          self.clip_length))
        item['joint_state_effort'] = torch.from_numpy(sensor_data_loader(self.data['joint_state_efforts'][video_id],
                                                          frame_start,
                                                          self.clip_length))

        item['base_velocity'] = torch.from_numpy(sensor_data_loader(self.data['base_velocities'][video_id],
                                                          frame_start,
                                                          self.clip_length))
        #Success 1 , failure 0
        if label == 1:
            is_not_anomalous = True
        else:
            is_not_anomalous = is_clip_anomalous(frame_start,
                                             self.clip_length,
                                             self.data['frame_labels'][video_id])
        item['frame_label'] = torch.from_numpy(np.array(is_not_anomalous*1.0,dtype=float))
        return item

    def __len__(self):
        return self.total_clips


# def main():
#     root_dir = '/scratch/sthodu2m/lucy/fdd/all_experiments'
#     dir_list = np.genfromtxt(os.path.join(root_dir, 'train_set.txt'), dtype=str)
#     d = HSRDataset(root_dir, dir_list, 18)
#     print('done loading!')
#     i = 0
#     for item in d:
#         print('shape of clip', item['clip'].shape)
#         print('shape of wrench', item['wrench'].shape)
#         print('shape of joint_state_velocity', item['joint_state_velocity'].shape)
#         print('frame lable', item['frame_label'])
#         print(type(item['clip']), type(item['frame_label']))
#         exit(0)
#
# if __name__ == '__main__':
#     main()
