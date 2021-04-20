import numpy as np
import sys

sys.path.insert(1, '../')
from constants import *
import random


def normalize_keypoints(keypoints, center_location):
    keypoints[np.isnan(keypoints)] = 0.
    # keypoints already in range[0, 1].

    # re-center
    center_broad = np.broadcast_to(center_location, keypoints.shape)
    keypoints -= center_broad
    
    keypoints[np.isnan(keypoints)] = 0.
    
    
 
  
    return keypoints


# ─── AUGMENTS ───────────────────────────────────────────────────────────────────


def augment(pose_frames, face_frames, left_hand_frames, right_hand_frames):
    # select augmentations.
    is_flip = np.random.random() < 0.5
    is_add_noise = np.random.random() < 0.5
    is_rotate = np.random.random() < 0.5
    is_zoom = np.random.random() < 0.75
    is_center_cut = np.random.random() < 0.75
    is_random_aspec_ratio = np.random.random() < 0.5

    if is_flip:
        pose_frames *= [-1, 1]
        face_frames *= [-1, 1]

        # swap hand.  
        left_hand_frames, right_hand_frames = right_hand_frames * [-1, 1], left_hand_frames * [-1, 1]

    if is_add_noise:
        pose_frames = add_noise(pose_frames, 0.07)

    if is_rotate:        
        rotate_degrees = random.randint(-20, 20)
        
        pose_frames = rotate_kp(pose_frames, rotate_degrees)
        face_frames = rotate_kp(face_frames, rotate_degrees)
        left_hand_frames = rotate_kp(left_hand_frames, rotate_degrees)
        right_hand_frames = rotate_kp(right_hand_frames, rotate_degrees)
        
    if is_zoom:
        zoom_m = 0.8 + 2*np.random.random()#was/2
        pose_frames *= zoom_m
        face_frames *= zoom_m
        left_hand_frames *= zoom_m
        right_hand_frames *= zoom_m
    
    if is_center_cut:
        cut_edge = 0.6 + np.random.random()/3     
        pose_frames[(abs(pose_frames)>cut_edge).any(axis=-1)] = 0.
        left_hand_frames[(abs(left_hand_frames)>cut_edge).any(axis=-1)] = 0.
        right_hand_frames[(abs(right_hand_frames)>cut_edge).any(axis=-1)] = 0.
        
        
    if is_random_aspec_ratio:
        ratio_x = 0.75+np.random.random()/2
        ratio_y = 0.75+np.random.random()/2
        pose_frames *= [ratio_x, ratio_y]
        face_frames *= [ratio_x, ratio_y]
        left_hand_frames *= [ratio_x, ratio_y]
        right_hand_frames *= [ratio_x, ratio_y]
        
        
    outmose_edge = 0.5
    pose_frames[(abs(pose_frames)>outmose_edge).any(axis=-1)] = 0.
    left_hand_frames[(abs(left_hand_frames)>outmose_edge).any(axis=-1)] = 0.
    right_hand_frames[(abs(right_hand_frames)>outmose_edge).any(axis=-1)] = 0.

   

    return pose_frames, face_frames, left_hand_frames, right_hand_frames


def add_noise(signal, factor=0.1):
    noise_std = np.std(signal) * np.random.random() * factor
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise


def rotate_kp(keypoints_xy_frames, rotate_degrees=10):
    """
    Random rotate keypoint coordinates by given angle.
    :param keypoints_xy_frames: keypoints coordinates.
    :param max_rotate: max rotation angle in degrees.
    :return:rotated keypoints coordinates
    """
    num_samples, num_kp, num_kp_dim = keypoints_xy_frames.shape

    keypoints_xy = keypoints_xy_frames.reshape(num_samples * num_kp, num_kp_dim)

    

    rotation_theta = np.radians(rotate_degrees)

    assert num_kp_dim == 2
    M_rotation = np.array([[np.cos(rotation_theta), -np.sin(rotation_theta)],
                           [np.sin(rotation_theta), np.cos(rotation_theta)]])

    keypoints_xyt = keypoints_xy.transpose()
    keypoints_out = M_rotation @ keypoints_xyt
    keypoints_out = keypoints_out.transpose()

    return keypoints_out.reshape(num_samples, num_kp, num_kp_dim)


# ─── SAMPLING METHODS ───────────────────────────────────────────────────────────

def uniform_sampling_indicies(total_frames, num_pick):
    tick = (total_frames - 2) / float(num_pick)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_pick)])
    return offsets + 1


def uniform_sampling(pose_frames, faces, left_hand, right_hand):
    """
    sampling frames by uniform selection.
    """
    indicies = uniform_sampling_indicies(total_frames=len(pose_frames), num_pick=NUM_FRAME_SAMPLES)

    return pose_frames[indicies], faces[indicies], left_hand[indicies], right_hand[indicies]


def random_sampling(pose_frames, faces, left_hand, right_hand):
    """
    sampling frames by random pick selection.
    """

    total_range = np.arange(len(pose_frames))
    np.random.shuffle(total_range)
    indicies = total_range[:NUM_FRAME_SAMPLES]
    indicies.sort()

    if len(indicies) < NUM_FRAME_SAMPLES:
        diff = NUM_FRAME_SAMPLES - len(indicies)
        pad = np.ones(diff) * indicies[-1]
        indicies = np.append(indicies, pad.astype('int'))

    return pose_frames[indicies], faces[indicies], left_hand[indicies], right_hand[indicies]
