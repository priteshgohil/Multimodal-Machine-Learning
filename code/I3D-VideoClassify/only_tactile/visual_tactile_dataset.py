import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torch.utils.data as data
from torchvision import transforms

import os
import os.path
import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from scipy import signal
from PIL import Image
import random
import matplotlib.pyplot as plt

# base class for dataloader vision.py


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


FILE_EXTENSIONS = ('front_0.jpg','tactile.txt','label.txt','front_rgb.mp4','left_rgb.mp4', 'pos.txt')
TACTILE_MAX_MAGNITUDE = 75431.01

def get_params(img, output_size):
    """random crop input sequence of frames.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    t, h, w, c = img.shape
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th) if h!=th else 0
    j = random.randint(0, w - tw) if w!=tw else 0
    return i, j, th, tw

def random_crop(imgs):
    i, j, h, w = get_params(a, (224,224))
    imgs = imgs[:, i:i+h, j:j+w, :]
    return imgs

def tactile_loader(path, frames):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    output =[] 
    TACTILE_TIME = 24
    UPSAMPLE_FREQ = 18
    up_samples=TACTILE_TIME*UPSAMPLE_FREQ
    with open(path, 'rb') as f:
        tactile_frame = pd.read_csv(f,delimiter=' ', header=None)
        tactile = tactile_frame.as_matrix()
        tactile = tactile.astype('float')
        tactile = signal.resample(tactile,up_samples)
        tactile[np.where(tactile<0.1)] = 0.  # remove all the negative samples with 0
        tactile = tactile/TACTILE_MAX_MAGNITUDE #normalize input in range of 0 to 1.
        for frame in frames:
            output.append(tactile[frame,:].reshape(4,4,1))
        return np.array(output)

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

def compress_image(cv_frame, size):
    """
    Args:   frame - cv image
            size - tuple of (W x H)
    """
    return cv2.resize(cv_frame, size)

def preprocess_image(cv_frame, size):
    """
    Args:   frame - cv image
            size - tuple of (W x H)
    """
    im = cv2.resize(cv_frame, size) #compress image
    im = np.array(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))) #convert to RGB image
    im = im/255 #normalize
    im = (im - 0.5)/0.5
    return im
    
def vid_loader(path):
    """
    1. Sample 64 frames from the video at equal interval.
    2. Compress image from (640 x 480) to (320 x 240)
    3.
    3. Use Random crop to get (224 x 224)
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    #print(path)
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    required_frames = 64
    frame_output = 5
    # 18x18 = 324  frames in total
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print("length",length)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
#     print("FPS", fps )
    duration = length / fps
#     print("duration",duration)

    images = []
    frames = []
    while success:
        if((count%5)==0 and (len(images)<required_frames)):
            image = preprocess_image(image, (320,240))
            images.append(image)
            frames.append(count)
        success,image = vidcap.read()
        count += 1
#     print("normal len : " ,len(images))
    missing_frames = required_frames - len(images)
#     print("missing frames : " ,missing_frames)
    if(missing_frames):
        remove_element = (missing_frames//frame_output)+1
        del images[-remove_element:]
        del frames[-remove_element:]
        vidcap = cv2.VideoCapture(path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, length-missing_frames-remove_element)
        success,image = vidcap.read()
        temp_count = 0
        while success:
            image = preprocess_image(image, (320,240))
            images.append(image)
            frames.append(length-missing_frames-remove_element+temp_count)
            success,image = vidcap.read()
            temp_count +=1
    while(len(images) is not required_frames):
        vidcap = cv2.VideoCapture(path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, length-1)
        success,image = vidcap.read()
        image = preprocess_image(image, (320,240))
        images.append(image)
        frames.append(length-1)
    return np.array(images),np.array(frames)

def preprocess_grayscale_image(cv_frame, size):
    """
    Args:   frame - cv image
            size - tuple of (W x H)
    """
    im = cv2.resize(cv_frame, size) #compress image
    im = im/255 #normalize
    im = (im - 0.5)/0.5
    return im

def flow_loader(path, frames):
    flow_x = []
    flow_y = []
    images = []
    images_x = []
    images_y = []
    count = 0
    required_frames = 64
    for img in os.listdir(path):
        if img.endswith('x.png'):
            flow_x.append(img)
        else:
            flow_y.append(img)
    flow_x = sorted(flow_x)
    flow_y = sorted(flow_y)
    for frame in frames:
        while True:
            if(frame==int(flow_x[count][6:10])):
                images_x.append(flow_x[count])
                images_y.append(flow_y[count])
                break
            else:
                count += 1
    for x,y in zip(images_x,images_y):
        img_x = cv2.imread(path + x, cv2.IMREAD_GRAYSCALE)
        img_x = preprocess_grayscale_image(img_x,(320,240))
        img_y = cv2.imread(path + y, cv2.IMREAD_GRAYSCALE)
        img_y = preprocess_grayscale_image(img_y,(320,240))
        images.append(np.stack((img_x,img_y)).transpose([1,2,0])) #output shape T x H x W x C
    if(len(images) is not required_frames):
        raise ValueError("missing flow frames")
    return np.array(images)
    
#     count = 0
#     required_frames = 64
#     frame_output = 5
#     length = len(flow_x)
#     print(flow_x)
#     flow_x_list = flow_x[::frame_output]
#     flow_y_list = flow_y[::frame_output]

#     missing_frames = required_frames - len(flow_x_list)
#     if(missing_frames>0):
#         remove_element = (missing_frames//frame_output)+1
#         del flow_x_list[-remove_element:]
#         del flow_y_list[-remove_element:]
#         missing_frames = required_frames - len(flow_x_list)
#         [flow_x_list.append(i) for i in flow_x[-missing_frames:]]
#         [flow_y_list.append(i) for i in flow_y[-missing_frames:]]
#     elif(missing_frames<0): #contains extra frames and need to remove
#         del flow_x_list[-abs(missing_frames):]
#         del flow_y_list[-abs(missing_frames):]
#     images = []    
#     for x,y in zip(flow_x_list,flow_y_list):
#         img_x = cv2.imread(path + x, cv2.IMREAD_GRAYSCALE)
#         img_x = preprocess_grayscale_image(img_x,(320,240))
#         img_y = cv2.imread(path + y, cv2.IMREAD_GRAYSCALE)
#         img_y = preprocess_grayscale_image(img_y,(320,240))
#         images.append(np.stack((img_x,img_y)).transpose([1,2,0])) #output shape T x H x W x C
#     return np.array(images)

def old_flow_loader(path):
    flow_x = []
    flow_y = []
    for img in os.listdir(path):
        if img.endswith('x.png'):
            flow_x.append(img)
        else:
            flow_y.append(img)
    flow_x = sorted(flow_x)
    flow_y = sorted(flow_y)
    count = 0
    required_frames = 64
    frame_output = 5
    length = len(flow_x)

    flow_x_list = flow_x[::frame_output]
    flow_y_list = flow_y[::frame_output]

    missing_frames = required_frames - len(flow_x_list)
    if(missing_frames>0):
        remove_element = (missing_frames//frame_output)+1
        del flow_x_list[-remove_element:]
        del flow_y_list[-remove_element:]
        missing_frames = required_frames - len(flow_x_list)
        [flow_x_list.append(i) for i in flow_x[-missing_frames:]]
        [flow_y_list.append(i) for i in flow_y[-missing_frames:]]
    elif(missing_frames<0): #contains extra frames and need to remove
        del flow_x_list[-abs(missing_frames):]
        del flow_y_list[-abs(missing_frames):]
    images = []    
    for x,y in zip(flow_x_list,flow_y_list):
        img_x = cv2.imread(path + x, cv2.IMREAD_GRAYSCALE)
        img_x = preprocess_grayscale_image(img_x,(320,240))
        img_y = cv2.imread(path + y, cv2.IMREAD_GRAYSCALE)
        img_y = preprocess_grayscale_image(img_y,(320,240))
        images.append(np.stack((img_x,img_y)).transpose([1,2,0])) #output shape T x H x W x C
    return np.array(images)
"""
Make dataset with following input arguments:
split_file : .txt file containing path to each modalities to be trained
root: dataset folder path
mode: decides how many samples we need in the output Ex: "RGB" or "V", "FLOW" or "F", "VF",
        "VFT", "VFTP" (video, flow, tactile, position)
"""
def make_dataset(split_file, root, mode, transform):
    dataset = []
    file_path = np.loadtxt(split_file,dtype=str)
    prefix = root.split(file_path[0][:32])[0]
    file_names = ['front_rgb.mp4','left_rgb.mp4','tactile.txt','pos.txt','label.txt','flow']
#     print(prefix+file_path[0]+file_names[0])
    for i,file in enumerate(file_path):
        front_video_path = prefix + file + file_names[0]
        left_video_path = prefix + file + file_names[1]
        tactile_path = prefix + file + file_names[2]
        pos_path = prefix + file + file_names[3]
        label_path = prefix + file + file_names[4]
        front_flow_path = prefix + file + file_names[5] + '/' + 'front_rgb/'
        left_flow_path = prefix + file + file_names[5] + '/' + 'left_rgb/'
        dataset.append((front_video_path, left_video_path, tactile_path, pos_path, \
                       front_flow_path, left_flow_path, np.loadtxt(label_path)[3]))
    return dataset



class VisualTactile(VisionDataset):
    """Dataloader customized to load Visual-Tactile dataset
    Args:
        split_file (string): path to .txt file where we have path to each test and train object.
        root (string): Root directory path.
        mode (string) : For future implementation. (Decide which data to load V, F, VF, VFT, VFTP)
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, split_file, root, mode=None, transform=None):
        super(VisualTactile, self).__init__(root)

        self.root = root
        self.mode = mode
        self.split_file = split_file
        self.transform = transform
        self.extensions = FILE_EXTENSIONS

        # classes = name of the folder
        # class_to_idx = dictionary with class name(folder name) and index
        """classes, class_to_idx = self._find_classes(self.root)"""
        self.samples = make_dataset(self.split_file, self.root, self.mode, self.transform)
#         self.samples = old_make_dataset(self.root, class_to_idx, extensions, is_valid_file)
#         print("Samaples" , self.samples[0])
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        """self.classes = classes
        self.class_to_idx = class_to_idx"""
        self.targets = [s[-1] for s in self.samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (f_video, l_video, tactile, pos, f_flow, l_flow, target) where target is successfull pickup or not.
        """
        #add conditoin here wich will return only the requested mode. Ex. V, or VFTP
        f_video, l_video, tactile, pos, f_flow, l_flow, target = self.samples[index]
        sample_f_video, selected_frames = vid_loader(f_video)
#         sample_l_video = vid_loader(l_video)
        sample_tactile = tactile_loader(tactile, selected_frames)
#         sample_pos = tactile_loader(pos)
        sample_target = target
#         sample_f_flow = flow_loader(f_flow, selected_frames)
#         sample_l_flow = flow_loader(l_flow)
        if self.transform is not None:
            sample_f_video = self.transform(sample_f_video)
            sample_l_video = self.transform(sample_l_video)
#         return sample_f_video, sample_l_video, sample_tactile, sample_pos, sample_target
#         return video_to_tensor(sample_f_video), video_to_tensor(sample_l_video), \
#                 torch.from_numpy(sample_tactile), torch.from_numpy(sample_pos), \
#                 video_to_tensor(sample_f_flow), video_to_tensor(sample_l_flow), \
#                 torch.from_numpy(np.array(sample_target))
#         return video_to_tensor(sample_f_video), video_to_tensor(sample_l_video), \
#                 torch.from_numpy(sample_tactile), torch.from_numpy(sample_pos), \
#                 torch.from_numpy(np.array(sample_target))
        return video_to_tensor(sample_tactile), sample_target

    def __len__(self):
        return len(self.samples)


class VisualTactileFolder(VisualTactile):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, split_file, root, mode=None, transform=None):
        super(VisualTactileFolder, self).__init__(split_file, root, mode=None, transform=None)
        self.imgs = self.samples

