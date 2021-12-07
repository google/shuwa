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

import numpy as np
import json
import tensorflow as tf
from classifier_utils import normalize_keypoints, uniform_sampling, random_sampling
import os, sys
sys.path.insert(1, '../')
from constants import *

module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/20211207.h5")


class ClassifierManager():
    def __init__(self):           
        self.model = tf.function(tf.keras.models.load_model(MODEL_PATH, compile=False))
          
        
    def preprocess_input(self, kp_stack, sampling_fn=uniform_sampling):                
        pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack = [np.stack(x) for x in kp_stack]
             
        # sampling             
        pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack = sampling_fn(pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack)                       
        
        # filter unuse keypoints.
        pose_kp_stack = pose_kp_stack[:, SELECTED_POSENET_JOINTS]
        face_kp_stack = face_kp_stack[:, SELECTED_FACE_JOINTS]


        # normalize
        nose_location = np.expand_dims(pose_kp_stack[:, POSENET_CENTER_INDEX].copy(), 1) # index=0
        midfin_location_l = np.expand_dims(left_hand_kp_stack[:, HAND_CENTER_INDEX].copy(), 1) # index=9
        midfin_location_r = np.expand_dims(right_hand_kp_stack[:, HAND_CENTER_INDEX].copy(), 1) # index=9
                
        pose_kp_stack = normalize_keypoints(pose_kp_stack, center_location=nose_location)
        face_kp_stack = normalize_keypoints(face_kp_stack, center_location=nose_location)
        left_hand_kp_stack = normalize_keypoints(left_hand_kp_stack, center_location=midfin_location_l)
        right_hand_kp_stack = normalize_keypoints(right_hand_kp_stack, center_location=midfin_location_r)    
        
        # expand dims.             
        pose_kp_stack = np.expand_dims(pose_kp_stack, 0)
        face_kp_stack = np.expand_dims(face_kp_stack, 0)
        left_hand_kp_stack = np.expand_dims(left_hand_kp_stack, 0)
        right_hand_kp_stack = np.expand_dims(right_hand_kp_stack, 0)
        
        return pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack

    
    def __call__(self, kp_stack):        
        net_input = self.preprocess_input(kp_stack)
        output_feats = self.model(net_input)
        
        return output_feats.numpy().reshape(-1)
  