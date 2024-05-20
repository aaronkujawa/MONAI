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

import torch

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric

__all__ = ["TrueFalsePosNegMetric", "TrueFalsePosNegHelper"]


class TrueFalsePosNegMetric(CumulativeIterationMetric):
    """
    Compute average Dice score for a set of pairs of prediction-groundtruth segmentations.

    It supports both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_pred` is expected to have binarized predictions and `y` can be single-channel class indices or in the
    one-hot format. The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background. `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]),
    `y` can also be in the format of `B1HW[D]`.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
        num_classes: number of input channels (always including the background). When this is None,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.

    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        num_classes: int | None = None,
        which: str = "true_pos",
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.num_classes = num_classes
        self.which = which
        self.trueFalsePosNegHelper = TrueFalsePosNegHelper(
            include_background=self.include_background,
            reduction=MetricReduction.NONE,
            get_not_nans=False,
            softmax=False,
            num_classes=self.num_classes,
            which=self.which,
        )

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean Dice metric. `y` can be single-channel class indices or
                in the one-hot format.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute true positives (BxC) for  each channel for each batch
        return self.trueFalsePosNegHelper(y_pred, y)

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction and aggregation logic for the output of `compute_dice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class TrueFalsePosNegHelper:
    """
    Compute Dice score between two tensors `y_pred` and `y`.
    `y_pred` and `y` can be single-channel class indices or in the one-hot format.

    Example:

    .. code-block:: python

        import torch
        from monai.metrics import DiceHelper

        n_classes, batch_size = 5, 16
        spatial_shape = (128, 128, 128)

        y_pred = torch.rand(batch_size, n_classes, *spatial_shape).float()  # predictions
        y = torch.randint(0, n_classes, size=(batch_size, 1, *spatial_shape)).long()  # ground truth

        score, not_nans = DiceHelper(include_background=False, sigmoid=True, softmax=True)(y_pred, y)
        print(score, not_nans)

    """

    def __init__(
        self,
        include_background: bool | None = None,
        sigmoid: bool = False,
        softmax: bool | None = None,
        activate: bool = False,
        get_not_nans: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN_BATCH,
        num_classes: int | None = None,
        which: str = "true_pos",
    ) -> None:
        """

        Args:
            include_background: whether to include the score on the first channel
                (default to the value of `sigmoid`, False).
            sigmoid: whether ``y_pred`` are/will be sigmoid activated outputs. If True, thresholding at 0.5
                will be performed to get the discrete prediction. Defaults to False.
            softmax: whether ``y_pred`` are softmax activated outputs. If True, `argmax` will be performed to
                get the discrete prediction. Defaults to the value of ``not sigmoid``.
            activate: whether to apply sigmoid to ``y_pred`` if ``sigmoid`` is True. Defaults to False.
                This option is only valid when ``sigmoid`` is True.
            get_not_nans: whether to return the number of not-nan values.
            reduction: define mode of reduction to the metrics
            ignore_empty: if `True`, NaN value will be set for empty ground truth cases.
                If `False`, 1 will be set if the Union of ``y_pred`` and ``y`` is empty.
            num_classes: number of input channels (always including the background). When this is None,
                ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
                single-channel class indices and the number of classes is not automatically inferred from data.
        """
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.include_background = sigmoid if include_background is None else include_background
        self.softmax = not sigmoid if softmax is None else softmax
        self.activate = activate
        self.num_classes = num_classes
        self.which = which

    def compute_channel(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """"""
        if self.which == "true_pos":
            return (y_pred * y).sum()
        elif self.which == "false_pos":
            return (y_pred * (1 - y)).sum()
        elif self.which == "false_neg":
            return ((1 - y_pred) * y).sum()
        elif self.which == "true_neg":
            return ((1 - y_pred) * (1 - y)).sum()
        else:
            raise ValueError(f"Unknown value for 'which' argument: {self.which}, available options are: " \
                                                        "'true_pos', 'false_pos', 'false_neg', 'true_neg'")

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            y_pred: input predictions with shape (batch_size, num_classes or 1, spatial_dims...).
                the number of channels is inferred from ``y_pred.shape[1]`` when ``num_classes is None``.
            y: ground truth with shape (batch_size, num_classes or 1, spatial_dims...).
        """
        _softmax, _sigmoid = self.softmax, self.sigmoid
        if self.num_classes is None:
            n_pred_ch = y_pred.shape[1]  # y_pred is in one-hot format or multi-channel scores
        else:
            n_pred_ch = self.num_classes
            if y_pred.shape[1] == 1 and self.num_classes > 1:  # y_pred is single-channel class indices
                _softmax = _sigmoid = False

        if _softmax:
            if n_pred_ch > 1:
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        elif _sigmoid:
            if self.activate:
                y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred > 0.5

        first_ch = 0 if self.include_background else 1
        data = []
        for b in range(y_pred.shape[0]):
            c_list = []
            for c in range(first_ch, n_pred_ch) if n_pred_ch > 1 else [1]:
                x_pred = (y_pred[b, 0] == c).float() if (y_pred.shape[1] == 1) else y_pred[b, c]
                x = (y[b, 0] == c).float() if (y.shape[1] == 1) else y[b, c]
                c_list.append(self.compute_channel(x_pred, x))
            data.append(torch.stack(c_list))
        data = torch.stack(data, dim=0).contiguous()  # type: ignore

        f, not_nans = do_metric_reduction(data, self.reduction)  # type: ignore
        return (f, not_nans) if self.get_not_nans else f
