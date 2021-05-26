import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from path import Path
import scipy
import subprocess
import librosa
#import face_alignment
from skimage import transform as tf
import shutil
import dlib
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from torch.utils.data import Dataset, DataLoader
from librosa.core import load
from pydub import AudioSegment
from pydub.utils import mediainfo
from torchvision.utils import save_image
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread
from scipy import signal
import natsort

import python_speech_features

class VideoDataset(Dataset):
    def __init__(self, train, image_width):
        self.audio_dir, self.frame_dir = Path.db_dir(train)

        self.resize_height = image_width
        self.resize_width = image_width
        self.q_levels = 256
        self.EPSILON = 1e-2
        self.audio_rate = 50000
        self.video_rate = 25
        self.audio_feat_len = 0.2
        self.audio_feat_samples = 10000
        self.conversion_dict = {'s16': np.int16, 's32': np.int32}

        self.audio_file_names, self.frame_folder_names, labels = [], [], []
        for label in os.listdir(self.audio_dir):  #label: s1, s10, s11
            for audio_file_name in os.listdir(os.path.join(self.audio_dir, label)):  #fname: folders in s1, s10, s11
                self.audio_file_names.append(os.path.join(self.audio_dir, label, audio_file_name))
                frame_folder_name = audio_file_name.split('.')[0]
                self.frame_folder_names.append(os.path.join(self.frame_dir, label, frame_folder_name))
                labels.append(label)

        assert len(labels) == len(self.audio_file_names) == len(self.frame_folder_names)
        print('Number of videos: {:d}'.format(len(self.audio_file_names)))

    def load_frames(self, file_dir):
        frames = natsort.natsorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = imread(frame_name)
            #frame = cv2.imread(frame_name)
            frame = np.array(imresize(frame, (self.resize_width, self.resize_height))).astype(np.float64)
            buffer[i] = frame

        return buffer

    def load_audio(self, audio_path):
        speech, sr = librosa.load(audio_path, sr=16000)
        speech = np.insert(speech, 0, np.zeros(1965))
        speech = np.append(speech, np.zeros(1965))
        mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

        ind = 3
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc)
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)

        return audio_feat_seq, input_mfcc

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            #frame -= np.array([[[90.0, 98.0, 102.0]]])
            frame = frame/255.
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))

    def __len__(self):
        return len(self.audio_file_names)

    def __getitem__(self, index):
        # Loading and preprocessing.
        audio_seq, mfcc_seq = self.load_audio(self.audio_file_names[index])

        frame_seq = self.load_frames(self.frame_folder_names[index])
        frame_seq = self.normalize(frame_seq)
        #frame_seq = self.to_tensor(frame_seq)

        return audio_seq, mfcc_seq, frame_seq

