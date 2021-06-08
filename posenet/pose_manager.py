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
from decode_pose import decode_single_pose

import sys
sys.path.insert(1, '../')
from constants import *

module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/midfin_posenet_v6.h5")
INPUT_SIZE = 257

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.int32([keypoint_coords[left], keypoint_coords[right]]))
    return results


class PoseManager():
    def __init__(self):
        self.model = tf.function(tf.keras.models.load_model(MODEL_PATH, compile=False))
        

    def preprocess_input(self, image_np):
        image_np = cv2.resize(image_np, (INPUT_SIZE, INPUT_SIZE))
        image_np = (image_np / 127.5) - 1.
        return np.expand_dims(image_np, 0)
    

    def postprocess_output(self, net_output):     
        heatmaps, offsets, displacement_fwd, displacement_bwd = net_output
        pose_scores, keypoint_scores, keypoint_coords = decode_single_pose(
            heatmaps.numpy().squeeze(axis=0),
            offsets.numpy().squeeze(axis=0),
            displacement_fwd.numpy().squeeze(axis=0),
            displacement_bwd.numpy().squeeze(axis=0),
            output_stride=OUTPUT_STRIDE, score_threshold=MIN_PART_SCORE
            )

        # swap to x,y
        temp = keypoint_coords[0][:, 0].copy()
        keypoint_coords[0][:, 0] = keypoint_coords[0][:, 1]
        keypoint_coords[0][:, 1] = temp
        return pose_scores, keypoint_scores, keypoint_coords
    
    
    def __call__(self, image):
        image.ndim == 3
        in_h, in_w, in_d = image.shape
        assert in_h == in_w
        ratio = in_h/INPUT_SIZE
        
        net_input = self.preprocess_input(image)
        net_output = self.model(net_input)
        pose_scores, keypoint_scores, keypoint_coords = self.postprocess_output(net_output)   
    
        return pose_scores, keypoint_scores, keypoint_coords * ratio
    

    def draw_keypoints(self, frame_rgb, pose_scores, keypoint_scores, keypoint_coords,
                          min_pose_score=MIN_POSE_SCORE, min_part_score=MIN_PART_SCORE):

        adjacent_keypoints = []
        cv_keypoints = []
        for ii, score in enumerate(pose_scores):
            if score < min_pose_score:
                continue

            new_keypoints = get_adjacent_keypoints(keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)

            adjacent_keypoints.extend(new_keypoints)

            for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
                if ks < min_part_score:
                    continue
                cv_keypoints.append(cv2.KeyPoint(kc[0], kc[1], 10. * ks))



        cv2.drawKeypoints(frame_rgb, cv_keypoints, outImage=frame_rgb, color=(255, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT )
        cv2.polylines(frame_rgb, adjacent_keypoints, isClosed=False, color=(20, 225, 160), thickness=3)

