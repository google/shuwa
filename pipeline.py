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
import gin
import numpy as np

from modules import holistic, translator

gin.parse_config_file('configs/holistic.gin')
gin.parse_config_file('configs/translator_inference.gin')
gin.parse_config_file('configs/utils.gin')


class Pipeline:

    def __init__(self):
        super().__init__()
        self.is_recording = True
        self.knn_records = []
        self.holistic_manager = holistic.HolisticManager()
        self.translator_manager = translator.TranslatorManager()

        self.reset_pipeline()

    def reset_pipeline(self):
        self.pose_history = []
        self.face_history = []
        self.lh_history = []
        self.rh_history = []

    def update(self, frame_rgb):
        h, w, _ = frame_rgb.shape
        assert h == w

        frame_res = self.holistic_manager(frame_rgb)

        # Return if not found person.
        if np.all(frame_res["pose_4d"] == 0.):
            return

        if self.is_recording:
            cv2.putText(frame_rgb, "Recording...", (10, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 1)

            self.pose_history.append(frame_res["pose_4d"])
            self.face_history.append(frame_res["face_3d"])
            self.lh_history.append(frame_res["lh_3d"])
            self.rh_history.append(frame_res["rh_3d"])
