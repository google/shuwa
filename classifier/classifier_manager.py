import numpy as np
import json
import tensorflow as tf
from classifier_utils import normalize_keypoints, uniform_sampling, random_sampling
import os, sys
sys.path.insert(1, '../')
from constants import *

module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/20210203_916.h5")


class ClassifierManager():
    def __init__(self):      
        self.labels_txt = list(LABELS.keys())
        self.model = tf.function(tf.keras.models.load_model(MODEL_PATH, compile=False))
          
        
    def preprocess_input(self, kp_stack, sampling_fn=uniform_sampling):                
        pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack = [np.stack(x) for x in kp_stack]
             
        # sampling             
        pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack = sampling_fn(pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack)                       
        
        # filter unuse keypoints.
        pose_kp_stack = pose_kp_stack[:, SELECTED_POSENET_JOINTS]
        face_kp_stack = face_kp_stack[:, SELECTED_FACE_JOINTS]
                
        # normalize
        nose_location = pose_kp_stack[:, POSENET_CENTER_INDEX].copy().reshape(pose_kp_stack.shape[0], 1, pose_kp_stack.shape[2])
        
        pose_kp_stack = normalize_keypoints(pose_kp_stack, center_location=nose_location)
        face_kp_stack = normalize_keypoints(face_kp_stack, center_location=nose_location)
        left_hand_kp_stack = normalize_keypoints(left_hand_kp_stack, center_location=nose_location)
        right_hand_kp_stack = normalize_keypoints(right_hand_kp_stack, center_location=nose_location)    
        
        # expand dims.             
        pose_kp_stack = np.expand_dims(pose_kp_stack, 0)
        face_kp_stack = np.expand_dims(face_kp_stack, 0)
        left_hand_kp_stack = np.expand_dims(left_hand_kp_stack, 0)
        right_hand_kp_stack = np.expand_dims(right_hand_kp_stack, 0)
        
        return pose_kp_stack, face_kp_stack, left_hand_kp_stack, right_hand_kp_stack

    
    def __call__(self, kp_stack):        
        net_input = self.preprocess_input(kp_stack)
        output_feats= self.model(net_input)
        
        return output_feats.numpy().reshape(-1)
  