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





LABELS = {
    "0_Idle":0,
    "Hksl_able_to":1,
    "Hksl_bear":2,
    "Hksl_bicycle-Jsl_bicycle":3,
    "Hksl_busy":4,
    "Hksl_carrot":5,
    "Hksl_chef":6,
    "Hksl_coffee":7,
    "Hksl_correct":8,
    "Hksl_dog":9,
    "Hksl_dragonfly":10,
    "Hksl_finish":11,
    "Hksl_flip_flops":12,
    "Hksl_fortune_teller":13,
    "Hksl_garden":14,
    "Hksl_ginger":15,
    "Hksl_good":16,
    "Hksl_grass":17,
    "Hksl_grey":18,
    "Hksl_hat-Jsl_cap":19,
    "Hksl_hearing_aid-Jsl_hearing_aid":20,
    "Hksl_high_heels":21,
    "Hksl_how_are_you":22,
    "Hksl_leisurely":23,
    "Hksl_let_it_go":24,
    "Hksl_lipstick":25,
    "Hksl_machine":26,
    "Hksl_matter":27,
    "Hksl_milk_tea":28,
    "Hksl_motorcycle-Jsl_motorcycle":29,
    "Hksl_obstruct":30,
    "Hksl_post_office":31,
    "Hksl_purple":32,
    "Hksl_read":33,
    "Hksl_shark":34,
    "Hksl_signature":35,
    "Hksl_soft_drink":36,
    "Hksl_special":37,
    "Hksl_strange":38,
    "Hksl_summer":39,
    "Hksl_sunglasses":40,
    "Hksl_supermarket":41,
    "Hksl_tiger":42,
    "Hksl_tomato":43,
    "Hksl_trouble":44,
    "Hksl_understand":45,
    "Hksl_watermelon":46,
    "Hksl_welcome":47,
    "Hksl_wine":48,
    "Hksl_winter-Jsl_winter":49,
    "Hksl_worry":50,
    "Hksl_write":51,
    "Hksl_yes":52,
    "Jsl_age":53,
    "Jsl_ahh":54,
    "Jsl_be_good_at":55,
    "Jsl_bird":56,
    "Jsl_brown":57,
    "Jsl_canteen":58,
    "Jsl_carrot":59,
    "Jsl_cat":60,
    "Jsl_department_store":61,
    "Jsl_dog":62,
    "Jsl_dragonfly":63,
    "Jsl_draught_beer":64,
    "Jsl_dream":65,
    "Jsl_earring":66,
    "Jsl_elevator":67,
    "Jsl_example":68,
    "Jsl_fly":69,
    "Jsl_geta":70,
    "Jsl_get_up":71,
    "Jsl_goldfish":72,
    "Jsl_gray":73,
    "Jsl_hello_with_one_hand":74,
    "Jsl_hello_with_two_hands":75,
    "Jsl_high_heels":76,
    "Jsl_hot_spring":77,
    "Jsl_illness":78,
    "Jsl_imagination":79,
    "Jsl_insect":80,
    "Jsl_japanese_radish":81,
    "Jsl_law":82,
    "Jsl_lemon":83,
    "Jsl_lie":84,
    "Jsl_milk_tea":85,
    "Jsl_orange_juice":86,
    "Jsl_oversleep":87,
    "Jsl_pine":88,
    "Jsl_programmer":89,
    "Jsl_rice_field":90,
    "Jsl_sports_athelets":91,
    "Jsl_strawberry":92,
    "Jsl_summer":93,
    "Jsl_swim":94,
    "Jsl_tell_say":95,
    "Jsl_unique_not_usual":96,
    "Jsl_watch":97,
    "Jsl_wine":98,
    "Jsl_yellow":99
}
NUM_CLASSES = len(LABELS.keys())
LABELS_NAME = list(LABELS.keys())
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
