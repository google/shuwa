import gin
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from modules import utils

D_MODEL = 96


def get_triu_indicies(batch_size: int, n_joints: int, n_frames: int) -> npt.ArrayLike:
    """Get half top-right of Euclidean matrix, with batch index. 

    Args:
        batch_size (int): Batch size
        n_joints (int): Num joints define matrix rows, cols.

    Returns:
        npt.ArrayLike: 3D array of indices for tf.gather_nd.
    """
    triu_idxs = np.array(np.triu_indices(n_joints))
    num_member = len(triu_idxs[0])
    triu_idxs = np.tile(triu_idxs, batch_size * n_frames).transpose()
    grid = np.mgrid[0:batch_size, 0:n_frames, 0:num_member].reshape(3, -1).transpose()
    return np.concatenate([grid, triu_idxs], axis=1)[:, [0, 1, 3, 4]]


def batch_cdist(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Compute Euclidean matrix of the input batch.

    Args:
        a (tf.Tensor): 3D Tensor [batch_size, n_joints, dim]
        b (tf.Tensor): 3D Tensor [batch_size, n_joints, dim]

    Returns:
        tf.Tensor: 3D Tensor [batch_size, n_joints, n_joints]
    """
    cdist_b = tf.expand_dims(a, 2) - tf.expand_dims(b, 3)
    return tf.sqrt(tf.reduce_sum(tf.square(cdist_b), axis=-1))


@gin.configurable
def cdist(input: tf.Tensor, gather_idxs: tf.Tensor, ignore_value: tf.Tensor) -> tf.Tensor:
    """Compute Euclidean matrix of the input batch with maskig.

    Args:
        input (tf.Tensor): Input tensor [batch_size, n_frames, n_joints, dim]
        gather_idxs (tf.Tensor): Top-right indices of Euclidean matrix.
        ignore_value (tf.Tensor): Replace missing joints with this value.

    Returns:
        tf.Tensor: Euclidean distance matrix.
    """
    batch_size = tf.shape(input)[0]
    n_frames = tf.shape(input)[1]

    # Mask out missing joints.
    mask = tf.not_equal(input, ignore_value)
    mask = tf.math.reduce_all(mask, axis=-1)
    # Convert to float for multiply.
    mask_float = tf.where(mask, 1., 0.)
    mask_mat = tf.expand_dims(mask_float, 2) * tf.expand_dims(mask_float, 3)
    # Back to bool matrix
    mask_mat = tf.math.equal(mask_mat, 1.)

    # distance matrix. b,f,15,15
    dist_mat = batch_cdist(input, input)

    # Apply mask.
    dist_mat = tf.where(mask_mat, dist_mat, ignore_value)

    # Gather only upper-right half of distance matrix.
    dist_mat = tf.gather_nd(dist_mat, gather_idxs, batch_dims=0)

    return tf.reshape(dist_mat, shape=[batch_size, n_frames, -1])


@gin.configurable
def poses_diff(x, ignore_value: tf.Tensor):
    mask = tf.not_equal(x, ignore_value)
    # mask_f = tf.cast(mask, 'int8')
    mask_f = tf.where(mask, 1., 0.)
    mask_f = mask_f[:, 1:, :, :] * mask_f[:, :-1, :, :]
    mask_f = tf.concat([tf.expand_dims(mask_f[:, 0, :, :], 1), mask_f], axis=1)
    # mask_b = tf.cast(mask_f, 'bool')
    mask_b = tf.math.equal(mask_f, 1.)

    x = x[:, 1:, :, :] - x[:, :-1, :, :]
    x_d = tf.expand_dims(x[:, 0, :, :], 1)
    x_d = tf.concat([x_d, x], axis=1)

    return tf.where(mask_b, x_d, ignore_value)


def pose_motion(raw_poses):
    batch_size = raw_poses.shape[0]
    diff_slow = poses_diff(raw_poses)
    # flatten last 2 dims.
    diff_slow = tf.reshape(diff_slow, (batch_size, diff_slow.shape[1], diff_slow.shape[2] * diff_slow.shape[3]))

    return diff_slow


##
def c1D(x, filters, kernel):
    x = Conv1D(filters, kernel_size=kernel, padding="same", kernel_regularizer="l2")(x)
    x = ELU()(x)
    return x


def d1D(x, filters):
    x = Dense(filters, kernel_regularizer="l2")(x)
    x = ELU()(x)
    return x


def conv_enc(filters, n_gather, n_frames):
    encoder_input = Input(shape=(n_frames, n_gather))
    x = c1D(encoder_input, filters * 2, 1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)
    x = Flatten()(x)
    return Model(inputs=encoder_input, outputs=x)


@gin.configurable
def get_model(batch_size: int, n_pose_feats: int, n_face_feats: int, n_hand_feats: int, n_classes: int, n_frames: int):

    # top-right Euclidean.
    gather_pose = get_triu_indicies(batch_size, n_joints=15, n_frames=n_frames)
    gather_face = get_triu_indicies(batch_size, n_joints=25, n_frames=n_frames)
    gather_hand = get_triu_indicies(batch_size, n_joints=21, n_frames=n_frames)

    # ------ INPUT --------
    pose_3d = Input(batch_shape=(batch_size, n_frames, 15, 3), name='pose_3d')
    face_3d = Input(batch_shape=(batch_size, n_frames, 25, 3), name='face_3d')
    lh_3d = Input(batch_shape=(batch_size, n_frames, 21, 3), name='lh_3d')
    rh_3d = Input(batch_shape=(batch_size, n_frames, 21, 3), name='rh_3d')

    # ------ PREPROCESS ------
    pose_3d_, face_3d_, lh_3d_, rh_3d_ = utils.skeleton_utils.preprocess_keypoints_tf(pose_3d, face_3d, lh_3d, rh_3d)

    # ======================= POSE ======================================
    pose_encoder = conv_enc(D_MODEL, n_gather=210, n_frames=n_frames)

    # dist-raw-diff_slow-diff_fast
    pose_dist = cdist(pose_3d_, gather_pose)
    pose_3d_f = tf.reshape(pose_3d_, [batch_size, n_frames, 15 * 3])
    pose_diff_slow = pose_motion(pose_3d_)

    # concat pose.
    pose_cat = tf.concat([pose_dist, pose_3d_f, pose_diff_slow], axis=-1)

    pose_enc = pose_encoder(pose_cat)
    pose_enc = d1D(pose_enc, 256)
    pose_feats = Dense(n_pose_feats)(pose_enc)
    pose_feats = tf.math.l2_normalize(pose_feats, axis=-1)

    # ======================= FACE ======================================
    face_encoder = conv_enc(D_MODEL, n_gather=325, n_frames=n_frames)

    face_dist = cdist(face_3d_, gather_face)
    face_enc = face_encoder(face_dist)
    face_enc = d1D(face_enc, 256)
    face_feats = Dense(n_face_feats)(face_enc)
    face_feats = tf.math.l2_normalize(face_feats, axis=-1)

    # ======================= HAND ======================================
    hand_encoder = conv_enc(D_MODEL, n_gather=294, n_frames=n_frames)

    # left hand.
    lh_dist = cdist(lh_3d_, gather_hand)
    lh_3d_f = tf.reshape(lh_3d_, [batch_size, n_frames, 21 * 3])
    lh_cat = tf.concat([lh_dist, lh_3d_f], axis=-1)
    lh_enc = hand_encoder(lh_cat)
    lh_enc = d1D(lh_enc, 256)
    lh_feats = Dense(n_hand_feats)(lh_enc)
    lh_feats = tf.math.l2_normalize(lh_feats, axis=-1)

    # right hand.
    rh_dist = cdist(rh_3d_, gather_hand)
    rh_3d_f = tf.reshape(rh_3d_, [batch_size, n_frames, 21 * 3])
    rh_cat = tf.concat([rh_dist, rh_3d_f], axis=-1)
    rh_enc = hand_encoder(rh_cat)
    rh_enc = d1D(rh_enc, 256)
    rh_feats = Dense(n_hand_feats)(rh_enc)
    rh_feats = tf.math.l2_normalize(rh_feats, axis=-1)

    # concat all
    feats_out = tf.concat([pose_feats, face_feats, lh_feats, rh_feats], axis=-1, name="feats_out")

    # ====================================== CLS OUTPUT ======================================
    cls_out = Dense(n_classes, activation=None, name="cls_out")(feats_out)

    model = Model(inputs=[pose_3d, face_3d, lh_3d, rh_3d], outputs=[feats_out, cls_out])

    return model
