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

import itertools
import os
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandAffined, LoadImaged, EnsureChannelFirstd, CropForegroundd, NormalizeIntensityd, \
    SampleForegroundLocationsd, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, \
    RandScaleIntensityFixedMeand, RandSimulateLowResolutiond, RandAdjustContrastd, RandFlipd, Compose, \
    AppendDownsampledd, CastToTyped, EnsureTyped, ConcatItemsd, DeleteItemsd
from tests.utils import assert_allclose, is_tf32_env
import nibabel as nib

_rtol = 1e-3 if is_tf32_env() else 1e-4

TESTS = []

def get_deep_supr_label_shapes(deep_supr_num, patch_size, strides):
    supr_label_shapes = [patch_size]
    for i in range(deep_supr_num):
        last_shape = supr_label_shapes[-1]
        curr_strides = strides[
            i + 1]  # ignore first set of strides, since they apply to downsampling prior to the first level
        downsampled_shape = [int(np.round(last / curr)) for last, curr in zip(last_shape, curr_strides)]
        supr_label_shapes.append(downsampled_shape)
    return supr_label_shapes


modality_keys = ["image_0000", "image_0001"]

# transform parameters
patch_size= [32, 64, 32]
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 1]]
deep_supr_num = 3
pos_sample_num = 2
neg_sample_num = 1
use_nonzero = True
use_prior = False

label_keys = ["label"]
all_keys = modality_keys + label_keys

# exclude the prior from the intensity transforms
mod_inty_keys = modality_keys[:-1] if use_prior else modality_keys

load_image = LoadImaged(keys=all_keys, image_only=True)
ensure_channel_first = EnsureChannelFirstd(keys=all_keys)
crop_transform = CropForegroundd(keys=all_keys, source_key=mod_inty_keys[0], start_coord_key=None, end_coord_key=None)

norm_transform = NormalizeIntensityd(keys=mod_inty_keys, nonzero=use_nonzero)
sample_foreground_locations = SampleForegroundLocationsd(label_keys=label_keys, num_samples=10000)

rand_affine = RandAffined(
    keys=all_keys,
    mode=(3,)*len(modality_keys) + ("nearest", ),  # 3 means third order spline interpolation
    prob=1.0,
    spatial_size=patch_size,
    rotate_range= (30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi),
    prob_rotate=0.2,
    translate_range=(0, 0, 0),
    foreground_oversampling_prob=pos_sample_num / neg_sample_num,
    label_key_for_foreground_oversampling="label",
    prob_translate=1.0,
    scale_range=((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)),
    prob_scale=0.2,
    padding_mode=("constant",)*len(modality_keys) + ("border", ),
)

rand_gauss_noise = RandGaussianNoised(keys=mod_inty_keys, std=0.1, prob=0.1)

rand_gauss_smooth = RandGaussianSmoothd(keys=mod_inty_keys,
                                        sigma_x=(0.5, 1.0),
                                        sigma_y=(0.5, 1.0),
                                        sigma_z=(0.5, 1.0),
                                        prob=0.2 * 0.5, )  # 0.5 comes from the per_channel_probability

scale_intensity = RandScaleIntensityd(keys=mod_inty_keys, factors=[-0.25, 0.25], prob=0.15)

shift_intensity = RandScaleIntensityFixedMeand(keys=mod_inty_keys, factors=[-0.25, 0.25], preserve_range=True,
                                               prob=0.15)

sim_lowres = RandSimulateLowResolutiond(keys=mod_inty_keys, prob=0.25*0.5, zoom_range=(0.5, 1.0))

adjust_contrast_inverted = RandAdjustContrastd(keys=mod_inty_keys, prob=0.1 * 1.0, gamma=(0.7, 1.5),
                                               invert_image=True, retain_stats=True)

adjust_contrast = RandAdjustContrastd(keys=mod_inty_keys, prob=0.3 * 1.0, gamma=(0.7, 1.5), invert_image=False,
                                      retain_stats=True)

mirror_x = RandFlipd(all_keys, spatial_axis=[0], prob=0.5)
mirror_y = RandFlipd(all_keys, spatial_axis=[1], prob=0.5)
mirror_z = RandFlipd(all_keys, spatial_axis=[2], prob=0.5)

supr_label_shapes = get_deep_supr_label_shapes(deep_supr_num, patch_size, strides)

transform = Compose([
    load_image,
    ensure_channel_first,
    crop_transform,
    norm_transform,  # -1
    sample_foreground_locations,  # 0
    rand_affine,  # 1
    rand_gauss_noise,  # 2
    rand_gauss_smooth,  # 3
    scale_intensity,  # 4
    shift_intensity,  # 5
    sim_lowres,  # 6
    adjust_contrast_inverted,  # 7
    adjust_contrast,  # 8
    mirror_x, mirror_y, mirror_z,  # 9
    AppendDownsampledd(label_keys, downsampled_shapes=supr_label_shapes),
    CastToTyped(keys=modality_keys, dtype=np.float32),
    EnsureTyped(keys=modality_keys),
    ConcatItemsd(keys=modality_keys, name="image", dim=0),
    DeleteItemsd(keys=modality_keys),
], unpack_items=True)

TESTS.append(
    [
        dict(
            transform=transform,
            keys=["image_0000", "image_0001", "label"],
            fnames=["test_image_0000.nii.gz",
                    "test_image_0001.nii.gz",
                    "test_label.nii.gz"],
            data=[np.arange(64 * 64 * 64).reshape((64, 64, 64)).astype(float),
                  np.arange(64 * 64 * 64).reshape((64, 64, 64)).astype(float),
                  (np.random.rand(64, 64, 64) > 0.25).astype(float)]
        )
    ]
)


class TestNnunetTransforms(unittest.TestCase):
    @parameterized.expand(x + [y] for x, y in itertools.product(TESTS, (True,)))
    def test_nnunet_transforms(self, input_params, track_meta):
        set_track_meta(track_meta)

        with tempfile.TemporaryDirectory() as tempdir:
            data_dict = {}
            for i in range(len(input_params['fnames'])):
                name = input_params['fnames'][i]
                key = input_params['keys'][i]
                data = input_params['data'][i]
                filename = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(data, np.eye(4)), filename)
                data_dict.update({key: filename})

            transform = input_params['transform']
            out = transform(data_dict)

        self.assertTrue("label" in out.keys())
        self.assertTrue("image" in out.keys())  # concatenation of image_000X ....
        self.assertEqual(list(out['image'].shape), [len(modality_keys)]+patch_size)  # concatenation has the right shape?
        self.assertEqual(len(out['label']), deep_supr_num + 1)  # list of tensor outputs (top level output plus deep supervision heads


if __name__ == "__main__":
    unittest.main()
