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
from parameterized import parameterized

from monai.transforms import AppendDownsampled
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for p in TEST_NDARRAYS:
    for val in [2, 3, -1, -2.5]:
        downsampled_shapes = [(5, 5, 5), (4, 4, 4)]
        TEST_CASES.append(
            [
                {"downsampled_shapes": downsampled_shapes},
                p(np.ones([10, 10, 9]) * val),
                [p(np.ones(s) * val) for s in downsampled_shapes],
                downsampled_shapes,
            ]
        )


class TestAppendDownsampled(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value_shape(self, input_param, img, out, expected_shapes):
        result = AppendDownsampled(**input_param)(img)

        self.assertEqual(len(result), len(expected_shapes))

        for res, o, exp_shape in zip(result, out, expected_shapes):
            self.assertTupleEqual(res.shape, exp_shape)
            assert_allclose(res, o, rtol=1e-3, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
