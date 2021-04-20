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

import numpy as np

joint_colors = np.array([[0.4, 0.4, 0.4],
                    [0.4, 0.0, 0.0],
                    [0.6, 0.0, 0.0],
                    [0.8, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.4, 0.4, 0.0],
                    [0.6, 0.6, 0.0],
                    [0.8, 0.8, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.4, 0.2],
                    [0.0, 0.6, 0.3],
                    [0.0, 0.8, 0.4],
                    [0.0, 1.0, 0.5],
                    [0.0, 0.2, 0.4],
                    [0.0, 0.3, 0.6],
                    [0.0, 0.4, 0.8],
                    [0.0, 0.5, 1.0],
                    [0.4, 0.0, 0.4],
                    [0.6, 0.0, 0.6],
                    [0.7, 0.0, 0.8],
                    [1.0, 0.0, 1.0]])*255

joint_colors = joint_colors[:, ::-1]




bones_colors = [((0, 1), joint_colors[1, :]),
            ((1, 2), joint_colors[2, :]),
            ((2, 3), joint_colors[3, :]),
            ((3, 4), joint_colors[4, :]),

            ((0, 5), joint_colors[5, :]),
            ((5, 6), joint_colors[6, :]),
            ((6, 7), joint_colors[7, :]),
            ((7, 8), joint_colors[8, :]),

            ((0, 9), joint_colors[9, :]),
            ((9, 10), joint_colors[10, :]),
            ((10, 11), joint_colors[11, :]),
            ((11, 12), joint_colors[12, :]),

            ((0, 13), joint_colors[13, :]),
            ((13, 14), joint_colors[14, :]),
            ((14, 15), joint_colors[15, :]),
            ((15, 16), joint_colors[16, :]),

            ((0, 17), joint_colors[17, :]),
            ((17, 18), joint_colors[18, :]),
            ((18, 19), joint_colors[19, :]),
            ((19, 20), joint_colors[20, :])]

