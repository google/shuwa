import math
import random

import gin
import numpy as np


def get_rx(deg):
    rad = math.radians(deg)
    s = math.sin(rad)
    c = math.cos(rad)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def get_ry(deg):
    rad = math.radians(deg)
    s = math.sin(rad)
    c = math.cos(rad)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def get_rz(deg):
    rad = math.radians(deg)
    s = math.sin(rad)
    c = math.cos(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@gin.configurable
def shift_pose(kps, l_shoulder, r_shoulder, max_shift, ignore_value):
    """
    Random shift hand part in pose keypoints.
    """

    mask = np.not_equal(kps, ignore_value)
    # bf, 21, 3
    assert len(kps.shape) == 3
    pose_shifted = kps.copy()

    # Calculate shift range.
    unit = np.linalg.norm(kps[:, l_shoulder, :] - kps[:, r_shoulder, :], axis=1)
    shift = (2 * (unit * max_shift) * np.random.random()) - (unit * max_shift)

    # Shift.
    shifted_hand = kps[:, 7:] + shift[:, np.newaxis, np.newaxis]
    pose_shifted[:, 7:] = shifted_hand
    pose_shifted = np.where(mask, pose_shifted, ignore_value)

    return pose_shifted


@gin.configurable
def random_rotate(kps, max_deg, root_idx, ignore_value):
    """
    Random rotate hand keypoints in xyz axes.
    """

    if np.all(kps == ignore_value):
        return kps

    mask = np.not_equal(kps, ignore_value)

    # bf, 21, 3
    assert len(kps.shape) == 3
    root = kps[:, root_idx].copy()[:, np.newaxis]

    rx = get_rx(random.randint(-max_deg, max_deg))
    ry = get_ry(random.randint(-max_deg, max_deg))
    rz = get_rz(random.randint(-max_deg, max_deg))

    kps -= root
    kps = kps @ rx @ ry @ rz
    kps += root
    kps = np.where(mask, kps, ignore_value)

    return kps


@gin.configurable
def rotate_fingers(kps, max_deg, ignore_value):
    """
    Random rotate fingers in xyz axes.
    """

    if np.all(kps == ignore_value):
        return kps

    assert len(kps.shape) == 3
    # bf, 21, 3
    ids = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

    for id in ids:

        root = kps[:, id[0]].copy()[:, np.newaxis]

        rx = get_rx(random.randint(-max_deg, max_deg))
        ry = get_ry(random.randint(-max_deg, max_deg))
        rz = get_rz(random.randint(-max_deg, max_deg))

        kps_ = kps.copy() - root
        kps_ = kps_ @ rx @ ry @ rz
        kps_ += root
        kps[:, id] = kps_[:, id]

    return kps


def augment_video(vid):
    # Pose.
    # vid["pose_frames"] = shift_pose(vid["pose_frames"], max_shift=0.1)
    vid["pose_frames"] = random_rotate(vid["pose_frames"], max_deg=10, root_idx=0)
    # Face.
    vid["face_frames"] = random_rotate(vid["face_frames"], max_deg=10, root_idx=0)
    # Left hand.
    vid["lh_frames"] = random_rotate(vid["lh_frames"], max_deg=10, root_idx=9)
    vid["lh_frames"] = rotate_fingers(vid["lh_frames"], max_deg=10)
    # Right hand.
    vid["rh_frames"] = random_rotate(vid["rh_frames"], max_deg=10, root_idx=9)
    vid["rh_frames"] = rotate_fingers(vid["rh_frames"], max_deg=10)

    return vid
