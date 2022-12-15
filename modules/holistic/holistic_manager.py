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

import mediapipe as mp
import numpy as np
import numpy.typing as npt

from . import utils


class HolisticManager():

    def __init__(self):
        self.detector = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                                       smooth_landmarks=False,
                                                       min_tracking_confidence=0.5,
                                                       model_complexity=1)

    def __call__(self, frame: npt.ArrayLike) -> dict:

        # Empty results.
        pose_4d = np.zeros([15, 4], dtype=np.float32)
        face_3d = np.zeros([25, 3], dtype=np.float32)
        lh_3d = np.zeros([21, 3], dtype=np.float32)
        rh_3d = np.zeros([21, 3], dtype=np.float32)

        # Run detector.
        frame.flags.writeable = False
        mp_results = self.detector.process(frame)
        frame.flags.writeable = True

        # Parse results.
        if mp_results.pose_landmarks is not None:
            pose_4d = utils.parse_landmarks(mp_results.pose_landmarks.landmark, get_visibility=True)
            # Remove unwanted joints.
            pose_4d = utils.filter_pose(pose_4d)

        if mp_results.face_landmarks is not None:
            face_3d = utils.parse_landmarks(mp_results.face_landmarks.landmark)
            # Remove unwanted joints.
            face_3d = utils.filter_face(face_3d)

        if mp_results.left_hand_landmarks is not None:
            lh_3d = utils.parse_landmarks(mp_results.left_hand_landmarks.landmark)

        if mp_results.right_hand_landmarks is not None:
            rh_3d = utils.parse_landmarks(mp_results.right_hand_landmarks.landmark)

        # Draw.
        utils.mp_draw(frame, mp_results)

        return {"pose_4d": pose_4d, "face_3d": face_3d, "lh_3d": lh_3d, "rh_3d": rh_3d}
