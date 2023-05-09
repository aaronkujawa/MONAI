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

from monai.transforms import SampleForegroundLocations
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for p in TEST_NDARRAYS:
    for num_samples in [20, 30]:
        TEST_CASES.append([{"num_samples": num_samples}, p(np.ones([10, 10, 9])), p(np.ones([10, 10, 9])), num_samples])


class TestSampleForegroundLocations(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value_shape(self, input_param, img, out, expected_num_samples):
        result = SampleForegroundLocations(**input_param)(img)

        # check if metadata entry has the expected length
        self.assertEqual(len(result.meta["foreground_sample_locations"]), expected_num_samples)

        # output tensor should remain unchanged
        assert_allclose(result, out, rtol=1e-3, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
