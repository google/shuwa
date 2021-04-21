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

# ─── POSENET ────────────────────────────────────────────────────────────────────
POSENET_INPUT_SIZE = 257
SELECTED_POSENET_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18]
NUM_SELECTED_POSENET_JOINTS = len(SELECTED_POSENET_JOINTS)
POSENET_JOINT_DIMS = 2
POSE_THRESHOLD = 0.5
MIN_POSE_SCORE = 0.
MIN_PART_SCORE = 0.35
POSENET_CENTER_INDEX = 0
OUTPUT_STRIDE = 16

# ─── FACE_MESH ───────────────────────────────────────────────────────────────────
FACE_THRESHOLD = 0.1
SELECTED_FACE_JOINTS = [78, 191, 80, 13, 310, 415, 308, 324, 318, 14, 88, 95,
                        107, 69, 105, 52, 159, 145,
                        336, 299, 334, 282, 386, 374]
NUM_SELECTED_FACE_JOINTS = len(SELECTED_FACE_JOINTS)
FACE_JOINT_DIMS = 2

FACE_CENTER_INDEX = 1

# ─── HAND_PIPELINE ───────────────────────────────────────────────────────────────
MIN_HAND_RECT_SIZE = 140

POSENET_MIDFIN_THRESHOLD = 0.35
HAND_LANDMARK_THRESHOLD = 0.

NUM_HAND_JOINTS = 21
HAND_JOINT_DIMS = 2

HAND_CENTER_INDEX = 9

# ─── CLASSIFIER ─────────────────────────────────────────────────────────────────
SPLIT_TRAIN_VAL = 0.2
NUM_START_FILTERS = 64
NUM_FRAME_SAMPLES = 16
IGNORE_VALUE = 0.



LABELS = [
    "0_Idle",
    "Hksl_able_to",    
    "Hksl_bicycle-Jsl_bicycle",
    "Hksl_busy",
    "Hksl_carrot",
    "Hksl_chef",
    "Hksl_coffee",
    "Hksl_correct",
    "Hksl_dog",
    "Hksl_dragonfly",
    "Hksl_finish",
    "Hksl_flip_flops",
    "Hksl_fortune_teller",
    "Hksl_garden",
    "Hksl_ginger",
    "Hksl_good",
    "Hksl_grass",
    "Hksl_grey",
    "Hksl_hat-Jsl_cap",
    "Hksl_hearing_aid-Jsl_hearing_aid",
    "Hksl_high_heels",
    "Hksl_how_are_you",
    "Hksl_leisurely",
    "Hksl_let_it_go",
    "Hksl_lipstick",
    "Hksl_machine",
    "Hksl_matter",
    "Hksl_milk_tea",
    "Hksl_motorcycle-Jsl_motorcycle",
    "Hksl_obstruct",
    "Hksl_panda",
    "Hksl_post_office",
    "Hksl_purple",
    "Hksl_read",
    "Hksl_shark",
    "Hksl_signature",
    "Hksl_soft_drink",
    "Hksl_special",
    "Hksl_strange",
    "Hksl_summer",
    "Hksl_sunglasses",
    "Hksl_supermarket",
    "Hksl_tiger",
    "Hksl_tomato",
    "Hksl_trouble",
    "Hksl_understand",
    "Hksl_watermelon",
    "Hksl_welcome",
    "Hksl_wine",
    "Hksl_winter-Jsl_winter",
    "Hksl_worry",
    "Hksl_write",
    "Hksl_yes",
    "Jsl_age",
    "Jsl_ahh",
    "Jsl_be_good_at",
    "Jsl_bird",
    "Jsl_brown",
    "Jsl_canteen",
    "Jsl_carrot",
    "Jsl_cat",
    "Jsl_department_store",
    "Jsl_dog",
    "Jsl_dragonfly",
    "Jsl_draught_beer",
    "Jsl_dream",
    "Jsl_earring",
    "Jsl_elevator",
    "Jsl_example",
    "Jsl_fly",
    "Jsl_geta",
    "Jsl_get_up",
    "Jsl_goldfish",
    "Jsl_grey",
    "Jsl_hello_with_one_hand",
    "Jsl_hello_with_two_hands",
    "Jsl_high_heels",
    "Jsl_hot_spring",
    "Jsl_illness",
    "Jsl_imagination",
    "Jsl_insect",
    "Jsl_japanese_radish",
    "Jsl_law",
    "Jsl_lemon",
    "Jsl_lie",
    "Jsl_milk_tea",
    "Jsl_orange_juice",
    "Jsl_oversleep",
    "Jsl_pine",
    "Jsl_programmer",
    "Jsl_rice_field",
    "Jsl_sports_athelets",
    "Jsl_strawberry",
    "Jsl_summer",
    "Jsl_swim",
    "Jsl_tell_say",
    "Jsl_unique_not_usual",
    "Jsl_watch",
    "Jsl_wine",
    "Jsl_yellow"
]
NUM_CLASSES = len(LABELS)

# ─── POSENET DECODER ────────────────────────────────────────────────────────────

POSENET_PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist"
    , "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle", "leftMidfin", "rightMidfin"
]

NUM_KEYPOINTS = len(POSENET_PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(POSENET_PART_NAMES)}

# for draw
CONNECTED_PART_NAMES = [
    ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"),
    ("rightElbow", "rightShoulder"),
    ("rightElbow", "rightWrist"), ("leftShoulder", "rightShoulder"), ("leftWrist", "leftMidfin"),
    ("rightWrist", "rightMidfin")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

# displacement.
POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"), ("leftWrist", "leftMidfin"), ("rightWrist", "rightMidfin")
]

PARENT_CHILD_TUPLES = [(PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN]
# for custom decode.
# parent_idx, displacement_idx (list idx is target_idx)
DISPLACEMENT_MAP = [[None, None], [0, 0], [0, 2], [1, 1], [2, 3], [0, 4], [0, 10], [5, 5], [6, 11], [7, 6], [8, 13]]
WRIST_INDICES = [9, 10]
FACE_INDICES = [0, 1, 2, 3, 4]
TRUSTED_IDX = [0, 1, 2, 3, 4, 5, 6]
