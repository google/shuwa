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

import gin
import mediapipe as mp
import numpy as np
import numpy.typing as npt

# Drawing spec.
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def parse_landmarks(obj, get_visibility=False) -> npt.ArrayLike:
    result = np.zeros([len(obj), 4]) if get_visibility else np.zeros([len(obj), 3])
    for i in range(len(obj)):
        if get_visibility:
            result[i] = obj[i].x, obj[i].y, obj[i].z, obj[i].visibility
        else:
            result[i] = obj[i].x, obj[i].y, obj[i].z
    return result


def mp_draw(frame, results):
    # Draw face
    mp_drawing.draw_landmarks(frame,
                              results.face_landmarks,
                              landmark_drawing_spec=drawing_spec,
                              connection_drawing_spec=mp.solutions.holistic.FACEMESH_TESSELATION)

    # Draw left hand.
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    # Draw right hand.
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)


@gin.configurable
def filter_pose(pose_4d: npt.ArrayLike, selected_joints) -> npt.ArrayLike:
    """
    Remove unused points.
    """
    assert pose_4d.shape[-2:] == (33, 4)
    if pose_4d.ndim == 2:
        return pose_4d[selected_joints]
    if pose_4d.ndim == 3:
        return pose_4d[:, selected_joints]


@gin.configurable
def filter_face(pose_3d: npt.ArrayLike, selected_joints) -> npt.ArrayLike:
    """
    Frame level filter.
    """
    assert pose_3d.shape == (468, 3)
    return pose_3d[selected_joints]
