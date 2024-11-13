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
A collection of dictionary-based wrappers around a collection of transforms that serve as interfaces for external tools
defined in :py:class:`monai.transforms.interface.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Mapping
from copy import deepcopy

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.interface.array import ANTsAffineRegistration, ANTsApplyTransform, BrainExtraction
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple, ensure_tuple_rep

__all__ = [
    "ANTsAffineRegistrationd",
    "ANTsAffineRegistrationD",
    "ANTsAffineRegistrationDict",
    "ANTsApplyTransformd",
    "ANTsApplyTransformD",
    "ANTsApplyTransformDict",
    "BrainExtractiond",
    "BrainExtractionD",
    "BrainExtractionDict",
]


class ANTsAffineRegistrationd(MapTransform):
    """
    Dictionary-based wrapper of :class:`monai_inference_from_nnunet_folder.custom_transforms_array
    .ANTsAffineRegistration`.
    Registers a nifti image with an affine transformation to a reference image and resamples at the registered image
    at the coordinates of the reference image. Subsequently, the registered image is saved to disk as a new nifti
    file and the path to the new file is returned. Advanced Normalization Tools (ANTs) is required for this
    transform.
    """

    def __init__(
        self,
        keys: KeysCollection,
        moving_img_key: str,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
        output_folder_path: str = None,
        save_path_key: str = None,
        template_path: str = None,
    ) -> None:
        """
        :param keys: keys of the corresponding items to be transformed. See also:
        :param moving_img_key: ANTs will affinely register this image to the template and apply the resulting affine
            transform to all tensors in "keys".
        :param meta_keys: explicitly indicate the key to store the corresponding metadata dictionary. the metadata is a
            dictionary object which contains: filename, original_shape, etc. it can be a sequence of string, map to the keys.
            if None, will try to construct meta_keys by key_{meta_key_postfix}.
        :param meta_key_postfix: if meta_keys is None, use key_{postfix} to store the metadata of the nifti image, default
            is meta_dict. The metadata is a dictionary object. For example, load nifti file for image, store the metadata into
            image_meta_dict.
        :param allow_missing_keys: don’t raise exception if key is missing.
        :param output_folder_path: Path to folder where registered nifti files and transforms are saved. This only has
            an effect if save_path_key is not None.
        :param save_path_key: Key to the save path of the final prediction. The registered image will be saved to a
            subfolder in the corresponding directory.
        :param template_path: Path to reference image for affine registration and resampling.
        """
        super().__init__(keys, allow_missing_keys)
        self.moving_img_key = moving_img_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

        assert (output_folder_path or save_path_key), "Either output_folder_path or save_path_key must be provided."
        self.output_folder_path = output_folder_path
        self.save_path_key = save_path_key

        self.template_path = template_path

        assert moving_img_key in keys, f"moving_image_key ({moving_img_key}) has to be in keys ({keys})..."
        self.moving_img_key_idx = list(keys).index(moving_img_key)

        self.ANTsAffineRegistration = ANTsAffineRegistration(template_path=template_path)
        self.ANTsApplyTransform = ANTsApplyTransform()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # run ANTs registration algorithm between moving_img_key image and template_path image to find affine transformation
        key = self.moving_img_key
        meta_key = self.meta_keys[self.moving_img_key_idx]
        meta_key_postfix = self.meta_key_postfix[self.moving_img_key_idx]

        meta_key = meta_key or f"{key}_{meta_key_postfix}"

        # get path to the original input image
        original_space_image_path = deepcopy(d[key])

        # define paths where ANTs should save the registered image and affine transform matrix file
        if not self.save_path_key:
            mni_space_folder = self.output_folder_path
        else:
            mni_space_folder = os.path.join(os.path.dirname(d[self.save_path_key]), "preprocessed", "registered", key)

        mni_space_img_filename = os.path.basename(original_space_image_path).replace(
            ".nii.gz", "_ANTsregistered.nii.gz"
        )
        mni_space_img_path = os.path.join(mni_space_folder, mni_space_img_filename)
        output_affine_path = mni_space_img_path.replace(".nii.gz", "_")

        # this transformation will create a new Nifti-file with a registered image and return the path to it
        d[key] = self.ANTsAffineRegistration(original_space_image_path, mni_space_img_path, output_affine_path)

        # store paths in metadata
        d[meta_key + "_original_space_image_path"] = original_space_image_path
        output_affine_path = (
            output_affine_path + "0GenericAffine.mat"
        )  # This last part of the path is always added by ANTs
        d[meta_key + "_affine_trfm_file_path"] = output_affine_path

        # apply the saved transform to all other keys
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if key == self.moving_img_key:
                continue  # the moving_img_key image has already been transformed

            meta_key = meta_key or f"{key}_{meta_key_postfix}"

            # get path to the original input image
            original_space_image_path = deepcopy(d[key])

            # define paths where ANTs should save the registered image and affine transform matrix file
            mni_space_folder = os.path.join(os.path.dirname(d[self.save_path_key]), "preprocessed", "registered", key)
            os.makedirs(mni_space_folder, exist_ok=True)
            mni_space_img_filename = os.path.basename(original_space_image_path).replace(
                ".nii.gz", "_ANTsregistered.nii.gz"
            )
            mni_space_img_path = os.path.join(mni_space_folder, mni_space_img_filename)

            # store paths in metadata
            d[meta_key + "_original_space_image_path"] = original_space_image_path
            d[meta_key + "_affine_trfm_file_path"] = output_affine_path

            # this transformation will create a new Nifti-file with a registered image and return the path to it
            d[key] = self.ANTsApplyTransform(
                input_file_path=original_space_image_path,  # Path to the input image
                affine_trfm_file_path=output_affine_path,  # saved above
                reference_image_path=self.template_path,  # reference for resampling operation
                output_file_path=mni_space_img_path,  # Path to registered/resampled image
                use_inverse_trfm=False,
            )
        return d


class ANTsApplyTransformd(MapTransform):
    """
    Dictionary-based wrapper of :class:`monai_inference_from_nnunet_folder.custom_transforms_array.ANTsApplyTransform`.
    Uses Advanced Normalization Tools (ANTs) to apply an ANTs affine transformation to a nifti file and subsequently
    resamples the input image in the space of a reference image. This transform is used to invert a registration
    operation performed with ANTsAffineRegistration.
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        :param keys: keys of the corresponding items to be transformed. See also:
        :param meta_keys: explicitly indicate the key to store the corresponding metadata dictionary. the metadata is a
            dictionary object which contains: filename, original_shape, etc. it can be a sequence of string, map to the keys.
            if None, will try to construct meta_keys by key_{meta_key_postfix}.
        :param meta_key_postfix: if meta_keys is None, use key_{postfix} to store the metadata of the nifti image, default
            is meta_dict. The metadata is a dictionary object. For example, load nifti file for image, store the metadata into
            image_meta_dict.
        :param allow_missing_keys: don’t raise exception if key is missing.
        """

        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

        self.ANTsApplyTransform = ANTsApplyTransform()

    def __call__(
            self,
            data: Mapping[Hashable, NdarrayOrTensor],
            input_file_path: str,
            output_file_path: str,
            use_inverse_trfm: bool,
    ) -> dict[Hashable, NdarrayOrTensor]:
        """
        :param data: data dictionary that contains meta information about the paths of the affine transformation file
            and the path of the original image which is used as a reference for the resampling of transformed image.
        :param input_file_path: Path to nifti image/segmentation to be transformed
        :param output_file_path: Path to transformed nifti image/segmentation
        :param use_inverse_trfm: whether to apply the inverse of the transform store din the meta-data
        :return: Path to transformed nifti image/segmentation
        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"

            # get the original image path
            aff_path = d[meta_key + "_affine_trfm_file_path"]
            ref_path = d[meta_key + "_original_space_image_path"]

            # this transformation will create a new Nifti-file with a registered image and return the path to it
            d[key] = self.ANTsApplyTransform(
                input_file_path=input_file_path,
                affine_trfm_file_path=aff_path,
                reference_image_path=ref_path,
                output_file_path=output_file_path,
                use_inverse_trfm=True,
            )
        return d


class BrainExtractiond(MapTransform):
    """
    Dictionary-based wrapper of :class:`monai_inference_from_nnunet_folder.custom_transforms_array
    .BrainExtraction`.
    Uses HD-BET to perform brain extraction.
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
        output_folder_path: str = None,
        save_path_key: str = None,
    ) -> None:
        """
        :param keys: keys of the corresponding items to be transformed. See also:
        :param meta_keys: explicitly indicate the key to store the corresponding metadata dictionary. the metadata is a
            dictionary object which contains: filename, original_shape, etc. it can be a sequence of string, map to the keys.
            if None, will try to construct meta_keys by key_{meta_key_postfix}.
        :param meta_key_postfix: if meta_keys is None, use key_{postfix} to store the metadata of the nifti image, default
            is meta_dict. The metadata is a dictionary object. For example, load nifti file for image, store the metadata into
            image_meta_dict.
        :param allow_missing_keys: don’t raise exception if key is missing.
        :param output_folder_path: Path to folder where brain-extracted images are saved. This only has an effect if
            save_path_key is not None.
        :param save_path_key: Key to the save path of the final prediction. The brain-extracted image will be saved to
            a subfolder in the corresponding directory.
        """
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

        assert (output_folder_path or save_path_key), "Either output_folder_path or save_path_key must be provided."
        self.output_folder_path = output_folder_path
        self.save_path_key = save_path_key

        self.brainExtraction = BrainExtraction()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"

            # get path to the original input image
            original_image_path = deepcopy(d[key])

            # define paths where HD-BET should save the stripped image
            if not self.save_path_key:
                stripped_images_folder = self.output_folder_path
                stripped_img_filename = os.path.basename(original_image_path).replace(".nii.gz", "_stripped.nii.gz")
                stripped_img_path = os.path.join(stripped_images_folder, stripped_img_filename)
            else:
                stripped_images_folder = os.path.join(os.path.dirname(d[self.save_path_key]), "preprocessed", "brain_extracted", key)
                stripped_img_filename = os.path.basename(original_image_path).replace(".nii.gz", "_stripped.nii.gz")
                stripped_img_path = os.path.join(stripped_images_folder, stripped_img_filename)

            # store paths in metadata
            d[meta_key + "_original_image_path"] = original_image_path

            # this transformation will create a new Nifti-file with a brain-extracted image and return the path to it
            d[key] = self.brainExtraction(original_image_path, stripped_img_path)

        return d


BrainExtractionD = BrainExtractionDict = BrainExtractiond
ANTsAffineRegistrationD = ANTsAffineRegistrationDict = ANTsAffineRegistrationd
ANTsApplyTransformD = ANTsApplyTransformDict = ANTsApplyTransformd
