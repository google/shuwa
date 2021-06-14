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
from constants import *


# ─── COMMON-UTILS ───────────────────────────────────────────────────────────────
def normalize_radians(angle):
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


def compute_rotation(point1, point2):
    radians = (np.pi / 2) - np.arctan2(-(point2[1] - point1[1]), point2[0] - point1[0])
    radians = normalize_radians(radians)
    return np.degrees(radians)


def crop_from_rect(frame_rgb, rect):
    rect_width = int(rect[1][0])
    rect_height = int(rect[1][1])

    dst_pts = np.array([[0, rect_height - 1],
                        [0, 0],
                        [rect_width - 1, 0],
                        [rect_width - 1, rect_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(cv2.boxPoints(rect), dst_pts)
    return cv2.warpPerspective(frame_rgb, M, (rect_width, rect_height))


# ─── FACE-UTILS ─────────────────────────────────────────────────────────────────
def get_face_rect_from_posenet(keypoint_coords, factor=1.8):
    # use face_center instead of nose. 
    l_ear = keypoint_coords[0][POSENET_PART_NAMES.index("leftEar")][:2]
    r_ear = keypoint_coords[0][POSENET_PART_NAMES.index("rightEar")][:2]
    face_center = (l_ear + r_ear) / 2

    # create face rect.
    ear2ear = l_ear - r_ear
    box_size = np.linalg.norm(ear2ear) * factor

    upper_forehead_direction = ear2ear[1], -ear2ear[0]
    face_angle = compute_rotation(point1=(0, 0), point2=upper_forehead_direction)

    return tuple(face_center), (box_size, box_size), face_angle


# ─── HAND-UTILS ─────────────────────────────────────────────────────────────────

def get_hand_rect_from_posenet(keypoint_coords, wrist_idx=9, minfin_idx=17):
    wrist = keypoint_coords[wrist_idx][:2]
    midfin = keypoint_coords[minfin_idx][:2]

    # create hand rect.
    middle_finder_direction = midfin - wrist
    box_size = max(MIN_HAND_RECT_SIZE, np.linalg.norm(middle_finder_direction) * 3.5)
    hand_angle = compute_rotation(point1=(0, 0), point2=middle_finder_direction)

    return tuple(midfin), (box_size, box_size), hand_angle


def build_rotation_matrix(angle_rads):
    cosA = np.cos(angle_rads)
    sinA = np.sin(angle_rads)
    return np.array([[cosA, -sinA], [sinA, cosA]])


def local2global_keypoints(local_keypoints, global_box):
    global_box_center = np.array(global_box[0])
    global_box_size = np.array(global_box[1])[0]
    global_box_angle = -np.radians(global_box[2])

    # Center at 0.
    local_keypoints_norm = local_keypoints - (global_box_size / 2)

    # Rotate keypoints.
    rot_mat = build_rotation_matrix(global_box_angle)
    local_keypoints_norm = local_keypoints_norm.dot(rot_mat)

    return global_box_center + local_keypoints_norm
