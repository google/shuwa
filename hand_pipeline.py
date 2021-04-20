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

import cv2
import numpy as np
from hand_landmark import HandManager
from utils import crop_from_rect, get_hand_rect_from_posenet, local2global_keypoints
from constants import *


class HandPipeline():       
    def __init__(self):        
        self.hand_manager = HandManager()        

     
    def __call__(self, frame_rgb, posenet_keypoints, keypoint_scores):
        # create empty rect, keypoints.
        self.hands_rects = [None, None]
        self.hands_keypoints = [np.ones([NUM_HAND_JOINTS, HAND_JOINT_DIMS])*IGNORE_VALUE,
                                np.ones([NUM_HAND_JOINTS, HAND_JOINT_DIMS])*IGNORE_VALUE]
        
        # parse one person only.
        assert posenet_keypoints.shape[0] == 1
        posenet_keypoints = posenet_keypoints[0]
        
        temp_rects = [None, None]        
        # left    
        if (keypoint_scores[0, PART_IDS["leftMidfin"]] > POSENET_MIDFIN_THRESHOLD) and (keypoint_scores[0, PART_IDS["leftWrist"]] > POSENET_MIDFIN_THRESHOLD):          
            left_hand_rect = get_hand_rect_from_posenet(posenet_keypoints,
                                                        wrist_idx=PART_IDS["leftWrist"],
                                                        minfin_idx=PART_IDS["leftMidfin"])
            temp_rects[0] = left_hand_rect
           
            
        # right
        if (keypoint_scores[0, PART_IDS["rightMidfin"]] > POSENET_MIDFIN_THRESHOLD) and (keypoint_scores[0, PART_IDS["rightWrist"]] > POSENET_MIDFIN_THRESHOLD):          
            right_hand_rect = get_hand_rect_from_posenet(posenet_keypoints,
                                                         wrist_idx=PART_IDS["rightWrist"],
                                                         minfin_idx=PART_IDS["rightMidfin"])
            temp_rects[1] = right_hand_rect
            
            
        if temp_rects != [None, None]:
            self.batch_hands_inference(frame_rgb, temp_rects)         
       

        return self.hands_rects, self.hands_keypoints                      

  
    
    def batch_hands_inference(self, frame_rgb, rects):
        
        batch_size = sum(x is not None for x in rects)
        hand_rgb_batch = []
                
        # crop hand rgb.
        for r in rects:
            if r is not None:
                hand_rgb_batch.append(crop_from_rect(frame_rgb, r))
        
        # predict list of images.
        hands_flags, local_hands_keypoints = self.hand_manager(hand_rgb_batch)                

        hand_idx = 0
        batch_idx = 0              
        for rect in rects:
            if (rect is not None) and (hands_flags[batch_idx] > HAND_LANDMARK_THRESHOLD):
                
                # use only 2D keypoints.             
                local_hand_keypoints = local_hands_keypoints[batch_idx, :, :HAND_JOINT_DIMS]                
            
                # Convert local keypoints to global keypoints.
                global_hand_keypoints = local2global_keypoints(local_hand_keypoints, rects[hand_idx])         
                
                
                self.hands_rects[hand_idx] = rects[hand_idx]
                self.hands_keypoints[hand_idx] = global_hand_keypoints             
                batch_idx += 1
            hand_idx += 1
                      
      
       
    def draw(self, frame_rgb, hand_rect, hand_keypoints):

        for rect, keypoints in zip(hand_rect, hand_keypoints):
            if rect is not None:
                cv2.drawContours(frame_rgb, [cv2.boxPoints(rect).astype("int")], 0, (128, 0, 255), 2)
                self.hand_manager.draw_keypoints(frame_rgb, keypoints) 
