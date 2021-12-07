# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import pickle
import random
import sys
sys.path.insert(1, '../')

import numpy as np
from classifier_utils import normalize_keypoints, augment, uniform_sampling, random_sampling
from tensorflow.python.keras.utils.data_utils import Sequence

from constants import *
from tqdm import tqdm

def softmax(x):    
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class DDDataGenerator(Sequence):

    def __init__(self, root_folder, batch_size=2, use_augment=True):   
     
        self.batch_size = batch_size
        self.all_labels = self.load_keypoints(root_folder)
        self.use_augment = use_augment
        
        for k in self.all_labels.keys():
            print(k, len(self.all_labels[k]))
            

    def __iter__(self):
        return self

    def __len__(self):
        count = 0
        for k in self.all_labels.keys():
            count += len(self.all_labels[k])
        return count // self.batch_size
             
    
    def load_keypoints(self, root_folder):
        
        # print("[INFO] Loading all label in to memory...")
        root_folder = os.path.normpath(root_folder)                
        label_paths = glob.glob(root_folder+'\\*')        
        
        all_labels = {}
        for folder in tqdm(label_paths):
            name_txt = folder.split(os.sep)[-1]
            file_pattern = os.path.join(folder, '*.pkl')
            all_pkl = glob.glob(file_pattern)
            
            samples = []
            for pkl in all_pkl:
                # load picle file.         
                with open(pkl, 'rb') as f:
                    pose_frames, face_frames, left_hand_frames, right_hand_frames = pickle.load(f)                

                samples.append([pose_frames, face_frames, left_hand_frames, right_hand_frames])                
            all_labels[name_txt] = samples
            
        return all_labels
    

    
    def random_train_sample(self):
        """
        random pick sample from the datset.
        """      
        random_class_name = random.choice(list(self.all_labels.keys()))
        label_idx = LABELS.index(random_class_name)
       
        pose_frames, face_frames, left_hand_frames, right_hand_frames = random.choice(self.all_labels[random_class_name])            
        assert len(pose_frames) > 12
        
        # sampling frames.
        sampling_method = random.choice([uniform_sampling, random_sampling])
        pose_frames, face_frames, left_hand_frames, right_hand_frames = sampling_method(pose_frames, face_frames, left_hand_frames, right_hand_frames)

        # normalize
        nose_location = np.expand_dims(pose_frames[:, POSENET_CENTER_INDEX].copy(), 1) # index=0
        midfin_location_l = np.expand_dims(left_hand_frames[:, HAND_CENTER_INDEX].copy(), 1) # index=9
        midfin_location_r = np.expand_dims(right_hand_frames[:, HAND_CENTER_INDEX].copy(), 1) # index=9
        
        pose_frames = normalize_keypoints(pose_frames, center_location=nose_location)
        face_frames = normalize_keypoints(face_frames, center_location=nose_location)
        left_hand_frames = normalize_keypoints(left_hand_frames, center_location=midfin_location_l)
        right_hand_frames = normalize_keypoints(right_hand_frames, center_location=midfin_location_r)


        # augment
        if self.use_augment:
            pose_frames, face_frames, left_hand_frames, right_hand_frames = augment(pose_frames,
                                                                                    face_frames,
                                                                                    left_hand_frames,
                                                                                    right_hand_frames)
            
        # filter unuse keypoints.
        pose_frames = pose_frames[:, SELECTED_POSENET_JOINTS]
        face_frames = face_frames[:, SELECTED_FACE_JOINTS]

        
        return  [pose_frames, face_frames, left_hand_frames, right_hand_frames], label_idx
        
    
       
    def __getitem__(self, item):

        pose_frames_batch = np.empty([self.batch_size, NUM_FRAME_SAMPLES, NUM_SELECTED_POSENET_JOINTS, POSENET_JOINT_DIMS], dtype=np.float32)
        face_frames_batch = np.empty([self.batch_size, NUM_FRAME_SAMPLES, NUM_SELECTED_FACE_JOINTS, FACE_JOINT_DIMS], dtype=np.float32)
        left_hand_frames_batch = np.empty([self.batch_size, NUM_FRAME_SAMPLES, NUM_HAND_JOINTS, HAND_JOINT_DIMS], dtype=np.float32)
        right_hand_frames_batch = np.empty([self.batch_size, NUM_FRAME_SAMPLES, NUM_HAND_JOINTS, HAND_JOINT_DIMS], dtype=np.float32)
        
        y_batch = np.zeros([self.batch_size, 1], dtype=np.float32)
        

        for i in range(self.batch_size):
        
            [pose_frames, face_frames, left_hand_frames, right_hand_frames], label_idx = self.random_train_sample()          
              
            pose_frames_batch[i] = pose_frames
            face_frames_batch[i] = face_frames
            left_hand_frames_batch[i] = left_hand_frames
            right_hand_frames_batch[i] = right_hand_frames     
                        
            
            y_batch[i] = label_idx
           
        return [pose_frames_batch, face_frames_batch, left_hand_frames_batch, right_hand_frames_batch],  y_batch


