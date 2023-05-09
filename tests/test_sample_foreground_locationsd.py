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

from monai.transforms import SampleForegroundLocationsd
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for p in TEST_NDARRAYS:
    for num_samples in [20, 30]:
        TEST_CASES.append(
            [
                {"label_keys": ["label"], "num_samples": num_samples},
                {"image": p(np.zeros([10, 10, 9])), "label": p(np.ones([10, 10, 9]))},
                {"image": p(np.zeros([10, 10, 9])), "label": p(np.ones([10, 10, 9]))},
                num_samples,
            ]
        )


class TestSampleForegroundLocationsd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value_shape(self, input_param, test_input, output, expected_num_samples):
        result = SampleForegroundLocationsd(**input_param)(test_input)
        assert_allclose(
            result["image"], output["image"], rtol=1e-3
        )  # "image" should not have been affected by transform
        assert_allclose(
            result["label"], output["label"], rtol=1e-3, type_test="tensor"
        )  # "label" should not have been affected by transform
        if "label" in result:
            self.assertEqual(len(result["label"].meta["foreground_sample_locations"]), expected_num_samples)

            # output tensor should remain unchanged
            # assert_allclose(result["image"], output["image"], rtol=1e-3, type_test="tensor")
            # assert_allclose(result["label"], output["label"], rtol=1e-3, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
