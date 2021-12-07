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
import sys
import numpy as np
from constants import *
from utils import local2global_keypoints, crop_from_rect, get_face_rect_from_posenet


sys.path.insert(1, 'posenet')
sys.path.insert(1, 'face_landmark')
sys.path.insert(1, 'hand_landmark')
sys.path.insert(1, 'classifier')

from pose_manager import PoseManager
from face_manager import FaceManager
from hand_pipeline import HandPipeline
from classifier_manager import ClassifierManager

class Pipeline:

    def __init__(self):
        super().__init__()
        self.pose_manager = PoseManager()
        self.face_manager = FaceManager()
        self.hand_pipeline = HandPipeline()
        self.classifier = ClassifierManager()
        self.reset_pipeline()
        self.is_recording = True
        
        
    def reset_pipeline(self):               
        self.pose_history = []
        self.face_history = []
        self.left_hand_history = []
        self.right_hand_history = []


    def update(self, frame_rgb):            
        frame_h, frame_w, _ = frame_rgb.shape
        assert frame_h == frame_w      

        # ─── POSENET ────────────────────────────────────────────────────────────────────
        person_score, keypoint_scores, posenet_keypoints = self.pose_manager(frame_rgb)
        if person_score > POSE_THRESHOLD:
            # ─── FACE-LANDMARK ───────────────────────────────────────────────────────────────           
            face_rect = get_face_rect_from_posenet(posenet_keypoints)
            face_rgb = crop_from_rect(frame_rgb, face_rect) 
            face_flag, local_face_keypoints = self.face_manager(face_rgb)
            
            
            if face_flag[0] > FACE_THRESHOLD:
                # use only person 0, 2D keypoints.
                local_face_keypoints = local_face_keypoints[:, :FACE_JOINT_DIMS]

                # Convert local keypoints to global keypoints.
                global_face_keypoints = local2global_keypoints(local_face_keypoints, face_rect)

                # ─── HAND-LANDMARK ───────────────────────────────────────────────────────────────    
                hands_rects, global_hands_keypoints = self.hand_pipeline(frame_rgb, posenet_keypoints, keypoint_scores)                

                # ─── DRAW ───────────────────────────────────────────────────────────────────────               
                cv2.drawContours(frame_rgb, [cv2.boxPoints(face_rect).astype("int")], 0, (0, 255, 0), 2)                
                self.pose_manager.draw_keypoints(frame_rgb, person_score, keypoint_scores, posenet_keypoints)
                self.face_manager.draw_keypoints(frame_rgb, global_face_keypoints)
                self.hand_pipeline.draw(frame_rgb, hands_rects, global_hands_keypoints)

                if self.is_recording:
                    cv2.putText(frame_rgb, "Recording...", (10, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 1)
                    self.pose_history.append(posenet_keypoints[0]/frame_h)
                    self.face_history.append(global_face_keypoints/frame_h)
                    self.left_hand_history.append(global_hands_keypoints[0]/frame_h)
                    self.right_hand_history.append(global_hands_keypoints[1]/frame_h)
                

    def run_classifier(self):
        """Run sign classifier.

        Returns:
            np.ndarray: 832D.
            pose [0:255] : 256D
            face[256:319] : 64D
            hand[320:831] : 512D
        """
        feats = self.classifier([self.pose_history, self.face_history, self.left_hand_history, self.right_hand_history])
        self.reset_pipeline()
        return feats
    
    
    def run_knn_classifier(self, k=3):    
        feats = self.run_classifier()
        distances_by_feats = np.square(self.database - feats)        
        distances_total = np.sum(distances_by_feats, axis=-1)    
      

        # top k nearst samples.      
        top_indices = np.argsort(distances_total)[:k]
        top_lables = self.labels[top_indices]
        
        
        # mode.       
        vals, counts = np.unique(top_lables, return_counts=True)
        index = np.argmax(counts)
        result_class_name = vals[index]       

        return result_class_name
        

        
        
    
