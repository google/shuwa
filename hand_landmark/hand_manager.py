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

import os 
import numpy as np
import cv2
import tensorflow as tf
from color_palette import joint_colors, bones_colors

module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/hand_landmark.h5")
INPUT_SIZE = 224

class HandManager():
    
    
    def __init__(self):     
        self.model = tf.function(tf.keras.models.load_model(MODEL_PATH, compile=False))
        
        
    def preprocess_input(self, image_list):        
        if isinstance(image_list, list):        
            net_input = np.zeros([len(image_list), INPUT_SIZE, INPUT_SIZE, 3])
            ratio = np.zeros([len(image_list)])
            for idx, im in enumerate(image_list):
                in_h, in_w, in_d = image_list[idx].shape
                assert in_h == in_w
                ratio[idx] = in_h/INPUT_SIZE
                net_input[idx] = (cv2.resize(im, (INPUT_SIZE, INPUT_SIZE)))/255
        
        else:
            print("[ERROR] Inputs must be np.ndarray[].")
            exit()    
                   
        return ratio.reshape(-1,1,1), net_input


    def postprocess_output(self, net_output):
        hands_flags, _, hands_keypoints = net_output   
        hands_keypoints = hands_keypoints.numpy().reshape(-1, 21, 3)
        return hands_flags.numpy(), hands_keypoints 
        
    
    def __call__(self, image_list):
        
        ratio, net_input = self.preprocess_input(image_list)
        
        net_output = self.model(net_input)           
        hands_flags, hands_keypoints = self.postprocess_output(net_output)
        
        return hands_flags, hands_keypoints * ratio
        

  
    def draw_keypoints(self, frame_rgb, hand_keypoints, linewidth=2):        
       
        coords_xy = []
        for k in hand_keypoints:
            coords_xy.append(tuple(k.astype('int')))         
                       
        for connection, color in bones_colors:              
            pt1 = coords_xy[connection[0]]
            pt2 = coords_xy[connection[1]]
            
            if np.any(np.array(pt1) < 0):
                continue
            if np.any(np.array(pt2) < 0):
                continue                       
            cv2.line(frame_rgb, pt1, pt2, color, linewidth, -1)
        
   
        for i, xy in enumerate(coords_xy):        
            cv2.circle(frame_rgb, xy, linewidth, joint_colors[i, :], -1)
            # cv2.putText(frame_rgb, str(i), xy, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)        
        return frame_rgb