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
"""
A collection of transforms that serve as interfaces for external tools
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from ants.internal import get_lib_fn

from monai.transforms.transform import Transform
from monai.utils import optional_import

hd_bet_run, _ = optional_import("HD_BET.run")

__all__ = ["ANTsAffineRegistration", "ANTsApplyTransform", "BrainExtraction"]


class ANTsAffineRegistration(Transform):
    """
    Registers a nifti image with an affine transformation to a reference image and resamples at the registered image
    at the coordinates of the reference image. Subsequently, the registered image is saved to disk as a new nifti
    file and the path to the new file is returned.
    """

    def __init__(self, template_path: str) -> None:
        """
        :param template_path: Path to the reference image, for example an MNI template
        """
        self.template_path = template_path

    def __call__(self, original_space_image_path: str, mni_space_img_path: str, output_affine_path: str) -> str:
        """
        :param original_space_image_path: Path to the original (unregistered) nifti image
        :param mni_space_img_path: Path to the registered (output) nifti image
        :param output_affine_path: Path to the final affine transformation file (.mat extension)
        :return: Path to the registered (output) nifti image (same as mni_space_img_path)
        """

        fixed_file_path = self.template_path
        moving_file_path = original_space_image_path
        output_moved_file_path = mni_space_img_path

        ants_output = f"[{output_affine_path},{output_moved_file_path}]"
        ants_initial_moving_transforms = f"[{fixed_file_path},{moving_file_path},1]"
        ants_metric1 = f"MI[{fixed_file_path},{moving_file_path},1,32,Regular,0.25]"
        ants_metric2 = f"MI[{fixed_file_path},{moving_file_path},1,32,Regular,0.25]"

        antsRegistration_args = [
            "--verbose", "0",
            "--dimensionality", "3",
            "--float", "1",
            "--output", ants_output,
            "--interpolation", "Linear",
            "--use-histogram-matching", "1",
            "--winsorize-image-intensities", "[0.005,0.995]",
            "--transform", "Rigid[0.1]",
            "--convergence", "[1000x500x250x100x0,1e-6,10]",
            "--shrink-factors", "12x8x4x2x1",
            "--smoothing-sigmas", "4x3x2x1x1vox",
            "--initial-moving-transform", ants_initial_moving_transforms,
            "--metric", ants_metric1,
            "--transform", "Affine[0.1]",
            "--metric", ants_metric2,
            "--convergence", "[1000x500x250x100x0,1e-6,10]",
            "--shrink-factors", "12x8x4x2x1",
            "--smoothing-sigmas", "4x3x2x1x1vox",
            "--random-seed", "1234",
        ]

        # print(f"antsRegistration args: {antsRegistration_args}")

        os.makedirs(os.path.dirname(output_moved_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_affine_path), exist_ok=True)

        print("run affine registration...")
        lib_fn = get_lib_fn("antsRegistration")
        if lib_fn is None:
            raise RuntimeError("ANTsRegistration binary not found. Please install ANTsPy.")
        lib_fn(antsRegistration_args)

        return output_moved_file_path


class ANTsApplyTransform(Transform):
    """
    Uses ANTsPy to apply a specified affine transformation to a nifti file and subsequently
    resamples the input image in the space of a reference image. This transform is used to invert a registration
    operation performed with ANTsAffineRegistration.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        input_file_path: str,
        affine_trfm_file_path: str,
        reference_image_path: str,
        output_file_path: str,
        use_inverse_trfm: bool,
    ) -> str:
        """
        :param input_file_path: Path to the input image
        :param affine_trfm_file_path: Path to the affine transformation file (.mat extension)
        :param reference_image_path: Path to the reference image for the resampling operation.
        :param output_file_path: Path to the transformed and resampled output image.
        :param use_inverse_trfm: whether to invert the affine transform or not

        :return: Path to the transformed and resampled output image (same as output_file_path)
        """


        use_inverse_trfm = 1 if use_inverse_trfm else 0

        antsApplyTransforms_args = [
            "-d", "3",
            "-r", reference_image_path,
            "-t", f"[{affine_trfm_file_path}, {use_inverse_trfm}]",
            "-n", "NearestNeighbor",
            "-i", input_file_path,
            "-o", output_file_path,
        ]

        print(f"apply {'inverse' if use_inverse_trfm else ''} ANTs transform to {input_file_path}...")

        lib_fn = get_lib_fn("antsApplyTransforms")
        if lib_fn is None:
            raise RuntimeError("ANTsApplyTransforms binary not found. Please install ANTsPy.")
        lib_fn(antsApplyTransforms_args)

        return output_file_path


class BrainExtraction(Transform):
    """
    Uses HD-BET to perform brain extraction
    """

    def __init__(self) -> None:
        """ """
        pass

    def __call__(self, original_image_path: str, stripped_img_path: str) -> str:
        """
        :param original_image_path: Path to the original un-stripped T1 image
        :param stripped_img_path: Path to the stripped image
        :return: Path to the stripped image (same as stripped_img_path)
        """

        os.makedirs(os.path.dirname(stripped_img_path), exist_ok=True)

        # define context manager to suppress the print statements inside hd_bet
        @contextlib.contextmanager
        def nostdout():
            save_stdout = sys.stdout
            sys.stdout = io.StringIO()
            yield
            sys.stdout = save_stdout

        print("run brain-extraction...")
        with nostdout():
            print(f"run brain-extraction on {original_image_path}...")
            print(f"output will be saved to {stripped_img_path}...")
            hd_bet_run.run_hd_bet([original_image_path], [stripped_img_path], mode="fast", do_tta=False, bet=True)

        return stripped_img_path
