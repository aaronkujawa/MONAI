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

from monai.transforms import BinarizeLabel
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.extend(
        [
            [{"orig_labels": [3, 2, 1],
              "target_labels": [[0, 1, 1], [0, 1, 0], [0, 0, 1]]},
             p([[[3, 1], [1, 2]]]),  # input data
             p([[[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[1., 1.], [1., 0.]]]),  # expected output
             ],
            [{"orig_labels": [3, 5, 8],
              "target_labels": [[0, 1, 1], [0, 1, 0], [0, 0, 1]]},
             p([[[3, 8], [8, 5]]]),  # input data
             p([[[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[1., 1.], [1., 0.]]]),  # expected output
             ],
            [
                {"orig_labels": [3, 5, 8],
                 "target_labels": [[0, 1, 1], [0, 1, 0], [0, 0, 1]]},
                p([[[[3], [5], [5], [8]]]]),
                p(
                    [[[[0.0], [0.0], [0.0], [0.0]]],
                     [[[1.0], [1.0], [1.0], [0.0]]],
                     [[[1.0], [0.0], [0.0], [1.0]]]]
                ),  # expected output
            ],
        ]
    )


class TestBinarizeLabel(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input_data, expected_value):
        result = BinarizeLabel(**input_param)(input_data)
        if isinstance(expected_value, torch.Tensor):
            assert_allclose(result, expected_value)
        else:
            np.testing.assert_equal(result, expected_value)
        self.assertTupleEqual(result.shape, expected_value.shape)


if __name__ == "__main__":
    unittest.main()
