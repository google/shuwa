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
import cv2
import os
import tensorflow as tf

module_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(module_path, "model/face_landmark.h5")
INPUT_SIZE = 192


class FaceManager():       
    def __init__(self):
        self.model = tf.function(tf.keras.models.load_model(MODEL_PATH, compile=False))
        
        
    def preprocess_input(self, image_np):        
        image_np = cv2.resize(image_np, (INPUT_SIZE, INPUT_SIZE)) 
        return image_np / 255.


    def postprocess_output(self, net_output):
        face_flag = 1 / (1 + np.exp(-net_output[1].numpy().reshape(1)))
        face_landmark = net_output[0].numpy().reshape(468, 3)   
        return face_flag, face_landmark
    
    
    def __call__(self, image):
        image.ndim == 3
        in_h, in_w, in_d = image.shape
        assert in_h == in_w
        ratio = in_h/INPUT_SIZE
        
        image = self.preprocess_input(image)
        net_input = np.expand_dims(image, 0)        
        net_output = self.model(net_input)
        face_flag, face_landmark = self.postprocess_output(net_output)
        
        return face_flag, face_landmark * ratio
        
        
    
    def draw_keypoints(self, cv_frame, keypoints):
        for idx in range(0, 467):
            cv2.circle(cv_frame, (int(keypoints[idx, 0]), int(keypoints[idx, 1])),
                        radius=0, color=(204, 0, 102), thickness=2)