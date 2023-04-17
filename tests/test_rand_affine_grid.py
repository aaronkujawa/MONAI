# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import RandAffineGrid
from tests.utils import TEST_NDARRAYS_ALL, assert_allclose, is_tf32_env

_rtol = 1e-1 if is_tf32_env() else 1e-4

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append([{"device": device}, {"grid": p(torch.ones((3, 3, 3)))}, p(np.ones((3, 3, 3)))])
        TESTS.append(
            [
                {"rotate_range": (1, 2), "translate_range": (3, 3, 3)},
                {"grid": p(torch.arange(0, 27).reshape((3, 3, 3)))},
                p(
                    np.array(
                        [[[17.7142, 19.8156, 21.9171],
                          [24.0185, 26.1199, 28.2214],
                          [30.3228, 32.4242, 34.5257]],

                         [[58.8789, 62.1901, 65.5013],
                          [68.8125, 72.1238, 75.4350],
                          [78.7462, 82.0574, 85.3686]],

                         [[18.0000, 19.0000, 20.0000],
                          [21.0000, 22.0000, 23.0000],
                          [24.0000, 25.0000, 26.0000]]]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                {"translate_range": (3, 3, 3), "device": device},
                {"spatial_size": (3, 3, 3)},
                np.array(
                    [[[[-0.6921, -0.6921, -0.6921],
                       [-0.6921, -0.6921, -0.6921],
                       [-0.6921, -0.6921, -0.6921]],

                      [[0.3079, 0.3079, 0.3079],
                       [0.3079, 0.3079, 0.3079],
                       [0.3079, 0.3079, 0.3079]],

                      [[1.3079, 1.3079, 1.3079],
                       [1.3079, 1.3079, 1.3079],
                       [1.3079, 1.3079, 1.3079]]],

                     [[[0.3168, 0.3168, 0.3168],
                       [1.3168, 1.3168, 1.3168],
                       [2.3168, 2.3168, 2.3168]],

                      [[0.3168, 0.3168, 0.3168],
                       [1.3168, 1.3168, 1.3168],
                       [2.3168, 2.3168, 2.3168]],

                      [[0.3168, 0.3168, 0.3168],
                       [1.3168, 1.3168, 1.3168],
                       [2.3168, 2.3168, 2.3168]]],

                     [[[-1.4614, -0.4614, 0.5386],
                       [-1.4614, -0.4614, 0.5386],
                       [-1.4614, -0.4614, 0.5386]],

                      [[-1.4614, -0.4614, 0.5386],
                       [-1.4614, -0.4614, 0.5386],
                       [-1.4614, -0.4614, 0.5386]],

                      [[-1.4614, -0.4614, 0.5386],
                       [-1.4614, -0.4614, 0.5386],
                       [-1.4614, -0.4614, 0.5386]]],

                     [[[1.0000, 1.0000, 1.0000],
                       [1.0000, 1.0000, 1.0000],
                       [1.0000, 1.0000, 1.0000]],

                      [[1.0000, 1.0000, 1.0000],
                       [1.0000, 1.0000, 1.0000],
                       [1.0000, 1.0000, 1.0000]],

                      [[1.0000, 1.0000, 1.0000],
                       [1.0000, 1.0000, 1.0000],
                       [1.0000, 1.0000, 1.0000]]]]
                ),
            ]
        )
        TESTS.append(
            [
                {"device": device, "rotate_range": (1.0, 1.0, 1.0), "shear_range": (0.1,), "scale_range": (1.2,)},
                {"grid": p(torch.arange(0, 108).reshape((4, 3, 3, 3)))},
                p(
                    np.array(
                        [[[[-30.7709, -30.5800, -30.3891],
                           [-30.1981, -30.0072, -29.8163],
                           [-29.6254, -29.4344, -29.2435]],

                          [[-29.0526, -28.8617, -28.6707],
                           [-28.4798, -28.2889, -28.0980],
                           [-27.9070, -27.7161, -27.5252]],

                          [[-27.3343, -27.1433, -26.9524],
                           [-26.7615, -26.5706, -26.3796],
                           [-26.1887, -25.9978, -25.8069]]],

                         [[[42.8536, 44.3798, 45.9061],
                           [47.4324, 48.9586, 50.4849],
                           [52.0111, 53.5374, 55.0636]],

                          [[56.5899, 58.1161, 59.6424],
                           [61.1686, 62.6949, 64.2211],
                           [65.7474, 67.2736, 68.7999]],

                          [[70.3261, 71.8524, 73.3786],
                           [74.9049, 76.4312, 77.9574],
                           [79.4837, 81.0099, 82.5362]]],

                         [[[29.3580, 30.0760, 30.7941],
                           [31.5121, 32.2301, 32.9481],
                           [33.6661, 34.3842, 35.1022]],

                          [[35.8202, 36.5382, 37.2563],
                           [37.9743, 38.6923, 39.4103],
                           [40.1283, 40.8464, 41.5644]],

                          [[42.2824, 43.0004, 43.7184],
                           [44.4365, 45.1545, 45.8725],
                           [46.5905, 47.3085, 48.0266]]],

                         [[[81.0000, 82.0000, 83.0000],
                           [84.0000, 85.0000, 86.0000],
                           [87.0000, 88.0000, 89.0000]],

                          [[90.0000, 91.0000, 92.0000],
                           [93.0000, 94.0000, 95.0000],
                           [96.0000, 97.0000, 98.0000]],

                          [[99.0000, 100.0000, 101.0000],
                           [102.0000, 103.0000, 104.0000],
                           [105.0000, 106.0000, 107.0000]]]]
                    )
                ),
            ]
        )


class TestRandAffineGrid(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_affine_grid(self, input_param, input_data, expected_val):
        g = RandAffineGrid(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        if "device" in input_data:
            self.assertEqual(result.device, input_data[device])
        assert_allclose(result, expected_val, type_test=False, rtol=_rtol, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
