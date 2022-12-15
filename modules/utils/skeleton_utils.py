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
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from scipy.stats import beta


@gin.configurable
def filter_visibility(vid_res: dict, bp_hand_thres: float, lh_inpose: list[int], rh_inpose: list[int],
                      ignore_value) -> list[npt.ArrayLike]:

    # the value that was set from holistic module.
    MISSING_VALUE = 0.

    # Remove low confidence joints.

    assert vid_res["pose_frames"].shape[2] == 4, "[ERROR] Missing visibility channel."

    # _____ 1. _____
    # Remove hand in pose4d if blazepose_visibility < threshold.
    lh_view = vid_res["pose_frames"][:, lh_inpose]
    rh_view = vid_res["pose_frames"][:, rh_inpose]

    missing_pl = np.all(vid_res["pose_frames"][:, lh_inpose][:, :, 3] < bp_hand_thres, axis=1)
    missing_pr = np.all(vid_res["pose_frames"][:, rh_inpose][:, :, 3] < bp_hand_thres, axis=1)

    lh_view[missing_pl] = ignore_value
    rh_view[missing_pr] = ignore_value

    # _____ 2. _____
    # Remove hand in pose3d if no hand-landmarks detected.
    lh_view[vid_res["lh_frames"][:, 0, 0] == MISSING_VALUE] = ignore_value
    rh_view[vid_res["rh_frames"][:, 0, 0] == MISSING_VALUE] = ignore_value

    # Assign.
    vid_res["pose_frames"][:, lh_inpose] = lh_view
    vid_res["pose_frames"][:, rh_inpose] = rh_view

    # _____ 3. _____
    # Replace 0. to -5.
    vid_res["lh_frames"][vid_res["lh_frames"] == MISSING_VALUE] = ignore_value
    vid_res["rh_frames"][vid_res["rh_frames"] == MISSING_VALUE] = ignore_value

    # Remove hand-landmarks if no hand in pose4d.
    vid_res["lh_frames"][missing_pl] = ignore_value
    vid_res["rh_frames"][missing_pr] = ignore_value

    # Remove visibility channel.
    vid_res["pose_frames"] = vid_res["pose_frames"][:, :, :3]

    return vid_res


## ─── SKELETON SAMPLING ───────────────────────────────────────────────────────────


def apply_resampling(vid_res: dict, indices: list[int]):
    vid_res["pose_frames"] = vid_res["pose_frames"][indices]
    vid_res["face_frames"] = vid_res["face_frames"][indices]
    vid_res["lh_frames"] = vid_res["lh_frames"][indices]
    vid_res["rh_frames"] = vid_res["rh_frames"][indices]
    vid_res["n_frames"] = len(indices)
    return vid_res


def uniform_sampling(n_frames: int, n_pick: int):
    tick = (n_frames - 2) / float(n_pick)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(n_pick)])
    return offsets + 1


def random_sampling(n_frames: int, n_pick: int):

    total_range = np.arange(n_frames)
    np.random.shuffle(total_range)
    indices = total_range[:n_pick]
    indices.sort()

    if len(indices) < n_pick:
        diff = n_pick - len(indices)
        pad = np.ones(diff) * indices[-1]
        indices = np.append(indices, pad.astype('int'))

    return indices


beta_distribution = beta(a=2.5, b=3)


def beta_sampling(n_frames: int, n_pick: int):
    total_range = np.arange(n_frames)
    range_norm = total_range / n_frames
    beta_p = beta_distribution.pdf(range_norm) + 1e-5
    beta_p = beta_p / beta_p.sum()

    replace = False if n_frames >= n_pick else True
    picked = np.random.choice(total_range, size=n_pick, replace=replace, p=beta_p)
    return np.sort(picked)


def get_clip_params(clip_a=0.6, clip_b=0.9, center=0.55):
    clip_percent = clip_a + np.random.random() * (clip_b - clip_a)
    shift_offset = clip_a / 2 + np.random.random() * (center - clip_a / 2)
    return clip_percent, shift_offset


def clipped_uniform_sampling(n_frames: int, n_pick: int):
    clip_percent, shift_offset = get_clip_params()
    n_clipped = int(n_frames * clip_percent)
    s_offset = int((n_frames - n_clipped) * shift_offset)
    indices = s_offset + uniform_sampling(n_clipped, n_pick)
    return indices


def clipped_random_sampling(n_frames: int, n_pick: int):
    clip_percent, shift_offset = get_clip_params()
    n_clipped = int(n_frames * clip_percent)
    s_offset = int((n_frames - n_clipped) * shift_offset)
    indices = s_offset + random_sampling(n_clipped, n_pick)
    return indices


def clipped_beta_sampling(n_frames: int, n_pick: int):
    clip_percent, shift_offset = get_clip_params()
    n_clipped = int(n_frames * clip_percent)
    s_offset = int((n_frames - n_clipped) * shift_offset)
    indices = s_offset + beta_sampling(n_clipped, n_pick)
    return indices


## ─── PREPROCESS ───────────────────────────────────────────────────────────


def normalize_keypoints(keypoints, center_location, a_idx, b_idx, ignore_value, add_visibility):
    # Mask valid.
    mask = tf.not_equal(keypoints, ignore_value)
    # Re-center
    keypoints -= tf.expand_dims(center_location, 2)
    # Unit range.
    unit = tf.norm(keypoints[:, :, a_idx, :] - keypoints[:, :, b_idx, :], axis=2)
    keypoints = tf.math.divide_no_nan(keypoints, unit[:, :, tf.newaxis, tf.newaxis])
    # Apply mask.
    keypoints = tf.where(mask, keypoints, ignore_value)

    # Add visibility dimension.
    if add_visibility:
        visibility = mask[:, :, :, :1]
        keypoints = np.concatenate([keypoints, visibility], axis=-1)

    return keypoints


@gin.configurable
def preprocess_keypoints_tf(pose,
                            face,
                            lhand,
                            rhand,
                            midfin,
                            l_shoulder,
                            r_shoulder,
                            l_eye,
                            r_eye,
                            hand_wrist,
                            ignore_value,
                            add_visibility=False):
    # find center.
    pose_center = pose[:, :, 0]
    face_center = face[:, :, 0]
    midfin_location_l = lhand[:, :, midfin]
    midfin_location_r = rhand[:, :, midfin]

    # # normalize.
    pose = normalize_keypoints(pose,
                               center_location=pose_center,
                               a_idx=l_shoulder,
                               b_idx=r_shoulder,
                               ignore_value=ignore_value,
                               add_visibility=add_visibility)
    face = normalize_keypoints(face,
                               center_location=face_center,
                               a_idx=l_eye,
                               b_idx=r_eye,
                               ignore_value=ignore_value,
                               add_visibility=add_visibility)
    lhand = normalize_keypoints(lhand,
                                center_location=midfin_location_l,
                                a_idx=midfin,
                                b_idx=hand_wrist,
                                ignore_value=ignore_value,
                                add_visibility=add_visibility)
    rhand = normalize_keypoints(rhand,
                                center_location=midfin_location_r,
                                a_idx=midfin,
                                b_idx=hand_wrist,
                                ignore_value=ignore_value,
                                add_visibility=add_visibility)

    # flip right hand.
    if add_visibility:
        rhand *= [-1, 1, 1, 1]
    else:
        rhand *= [-1, 1, 1]

    return [pose, face, lhand, rhand]
