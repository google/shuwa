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


def crop_square(image):
    height, width, _ = image.shape

    if height < width:
        start_x = width // 2 - height // 2
        end_x = width // 2 + height // 2
        image = image[:, start_x:end_x]

    elif width < height:
        start_y = height - width
        image = image[start_y:, :]

    return image


def letterbox_image(image, size):
    iw, ih = image.shape[0:2][::-1]
    w, h = size, size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size, size, 3), np.uint8)
    new_image.fill(128)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy:dy + nh, dx:dx + nw, :] = image
    return new_image
