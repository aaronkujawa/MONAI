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

from monai.transforms import ResampleMultilabel
from monai.transforms.utils import create_grid
from tests.utils import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for q in TEST_NDARRAYS_ALL:
        for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
            # this test demonstrates the use of multilabel resampling.
            # the image "img" is a 1x2x2 image with 2 labels.  # C x H x W
            # one pixel has the label 5, and the other 3 pixels have the label 30.
            # the grid is a 1x1x1 grid (H x W x 1, where the 1 is the homogenous coordinate), therefore we sample
            # a single location from the image and the output will be a 1x1x1 image (C x H x W).
            # the grid is set to [-0.1, -0.1, 1] which is close to the center between the 4 pixels of the image, but
            # moved slightly to the left and up, towards the pixel labeled 5.
            # using the multi-label resampling with linear interpolation, the output is 30, even though the nearest
            # neighbor is 5. Because there are 3 pixels with label 30, and only 1 pixel with label 5, the linear
            # interpolation still favors the 30 label.
            # Resampling with the normal Resample class would have returned the label value 21 instead, since it does
            # not interpolate on a per-label basis.
            TESTS.append(
                [
                    dict(
                        padding_mode="zeros",
                        device=device,
                        mode="linear",
                    ),
                    {"grid": p([[[-0.1]],
                                [[-0.1]],
                                [[1.]]]),
                     "img": q([[[5, 30],
                                [30, 30]]])},
                    q(
                        np.array(
                            [[[30.0]]]
                        )
                    ),
                ]
            ),
            # the following tests are copied from test_resampler.py
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {"grid": p(create_grid((2, 2))), "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(np.array([[[0.0, 1.0], [2.0, 3.0]]])),
                ]
            )
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {"grid": p(create_grid((4, 4))), "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(
                        np.array(
                            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
                        )
                    ),
                ]
            )
            TESTS.append(
                [
                    dict(padding_mode="border", device=device),
                    {"grid": p(create_grid((4, 4))), "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(
                        np.array(
                            [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3, 3.0], [2.0, 2.0, 3.0, 3.0]]]
                        )
                    ),
                ]
            )
            # TESTS.append(  # not well defined nearest + reflection resampling
            #     [
            #         dict(padding_mode="reflection", device=device),
            #         {"grid": p(create_grid((4, 4))), "img": q(np.arange(4).reshape((1, 2, 2))), "mode": "nearest"},
            #         q(
            #             np.array(
            #                 [[[3.0, 2.0, 3.0, 2.0], [1.0, 0.0, 1.0, 0.0], [3.0, 2.0, 3.0, 2.0], [1.0, 0.0, 1.0, 0.0]]]
            #             )
            #         ),
            #     ]
            # )
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {
                        "grid": p(create_grid((4, 4, 4))),
                        "img": q(np.arange(8).reshape((1, 2, 2, 2))),
                        "mode": "bilinear",
                    },
                    q(
                        np.array(
                            [
                                [
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 2.0, 3.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 4.0, 5.0, 0.0],
                                        [0.0, 6.0, 7.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                ]
                            ]
                        )
                    ),
                ]
            )
            TESTS.append(
                [
                    dict(padding_mode="border", device=device),
                    {
                        "grid": p(create_grid((4, 4, 4))),
                        "img": q(np.arange(8).reshape((1, 2, 2, 2))),
                        "mode": "bilinear",
                    },
                    q(
                        np.array(
                            [
                                [
                                    [
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                    ],
                                    [
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                    ],
                                    [
                                        [4.0, 4.0, 5.0, 5.0],
                                        [4.0, 4.0, 5.0, 5.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                    ],
                                    [
                                        [4.0, 4.0, 5.0, 5.0],
                                        [4.0, 4.0, 5.0, 5.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                    ],
                                ]
                            ]
                        )
                    ),
                ]
            ),
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {"grid": p(create_grid((4, 4))),
                     "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(
                        np.array(
                            [[[0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 2.0, 3.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]]
                        )
                    ),
                ]
            )


class TestResampleMultilabel(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_resample_multilabel(self, input_param, input_data, expected_val):
        g = ResampleMultilabel(**input_param)
        result = g(**input_data)
        if "device" in input_data:
            self.assertEqual(result.device, input_data["device"])
        assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
