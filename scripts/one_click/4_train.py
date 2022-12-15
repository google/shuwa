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

import argparse
import json
import logging
from pathlib import Path

import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from modules import translator

gin.parse_config_file('configs/translator_train.gin')
gin.parse_config_file('configs/utils.gin')
logging.basicConfig(level=logging.DEBUG)

target_epoch = 100
steps_per_epoch = 500
n_hards = 50


def main(skeleton_dir, checkpoint):

    # ------------------------------- UPDATE LABELS ------------------------------ #
    LABELS = gin.config._CONFIG[('LABELS', 'gin.macro')]['value']
    N_CLASSES = len(LABELS.keys())
    h5_glosses = [p.stem for p in Path(skeleton_dir).glob("*.h5")]
    for g in h5_glosses:
        if g in LABELS.keys():
            continue
        else:
            logging.info(f"Found new gloss: {g}")
            g_idx = N_CLASSES
            LABELS[g] = [g_idx, g]
            N_CLASSES += 1
    with open("configs/labels.gin", "w") as f:
        dump_dict = json.dumps(LABELS, indent=0)
        f.writelines(f"LABELS = {dump_dict}\n")
        f.writelines(f"N_CLASSES = {N_CLASSES}")
    gin.parse_config_file('configs/translator_train.gin')
    gin.parse_config_file('configs/utils.gin')

    # ------------------------------- CREATE MODEl ------------------------------- #
    model = translator.get_model()
    batch_size = model.outputs[0].shape[0]
    n_feats = model.outputs[0].shape[1]
    n_classes = model.outputs[1].shape[1]
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"n_feats: {n_feats}")
    logging.info(f"n_classes: {n_classes}")
    model.load_weights(checkpoint)

    # ------------------------------ DATA GENERATOR ------------------------------ #
    train_generator = translator.DataGenerator(skeleton_dir)
    assert len(train_generator.labels_dict) == n_classes

    # --------------------------------- TRAINING --------------------------------- #
    optimizer = tf.optimizers.Adam(1e-3)

    acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    cce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

    initial_epoch = 0
    hards = None

    @tf.function
    def custom_train_step(inputs, y_true):
        with tf.GradientTape() as tape:
            feats_pred, cls_pred = model(inputs, training=True)

            cls_loss = cce(y_true, cls_pred)

        grads = tape.gradient(cls_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc_metrics.update_state(y_true, cls_pred)

        return cls_loss

    for ep in range(initial_epoch, target_epoch):
        acc_metrics.reset_states()

        for step in range(steps_per_epoch):
            inputs, y_true = train_generator.__getitem__(0, hards)
            cls_loss = custom_train_step(inputs, y_true)
            cls_loss_np = cls_loss.numpy()

            # Online Hard Mining
            hards_b = np.argsort(cls_loss_np)[-n_hards:]
            hards = y_true[hards_b].squeeze().tolist()
            print(" " * 100, end='\r')
            print(
                f"epoch-{ep:02d} step-{step} cls_loss-{np.mean(cls_loss_np):.4f} acc-{acc_metrics.result().numpy():.4f}",
                end='\r')

        if ep % 5 == 0:
            filepath = f"train_ckpts/{ep:02d}_{acc_metrics.result().numpy():.3f}.h5"
            model.save_weights(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('skeleton_dir')
    parser.add_argument('--checkpoint', default=None, required=False, help="Continue training from checkpoint.")
    args = parser.parse_args()

    skeleton_dir = Path(args.skeleton_dir)

    main(skeleton_dir, args.checkpoint)
