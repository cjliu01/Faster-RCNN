import torch
from torch import nn
from torchvision.ops import roi_align, roi_pool


class MultiScaleRoIOperation(nn.Module):
    def __init__(self, featmap_names, output_size, sampling_ratio=None, method='align'):
        super(MultiScaleRoIOperation, self).__init__()
        assert all([type(featmap_name) == type('') for featmap_name in featmap_names]), \
            'featmap_name must be a str type.'
        self.method = method
        if method == 'align':
            assert type(sampling_ratio) in [int, float], 'sampling_ratio must be int or float when method is align'
            self.sampling_ratio = sampling_ratio
        self.featmap_names = featmap_names
        self.output_size = output_size

    def forward(self, features, rois, image_shape):
        """
        Arguments:
            features (Dict[Tensor]): FPN feature maps
            rois (List[Tensor[N, 4]]): proposal boxes
            image_shape (Tuple[H, W]): image shape
        Returns:
            Tensor:
                Pooled features
        """
        filtered_features = self._filter_inputs(features, self.featmap_names)
        rois = self._convert_to_roi_format(rois)

        scales = self._setup_scales(filtered_features, image_shape)

        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

        k0 = 4
        box_area = (rois[:, 4] - rois[:, 2]) * (rois[:, 3] - rois[:, 1])
        k = torch.floor(k0 + torch.log2(torch.sqrt(box_area) / 224))
        k = torch.clamp(k, lvl_min, lvl_max) - lvl_min

        all_level_pooled_feature = torch.zeros([len(rois), filtered_features[0].shape[1], *self.output_size],
                                               dtype=filtered_features[0].dtype, device=rois.device)
        for idx, (featmap_in_level, scale) in enumerate(zip(filtered_features, scales)):
            mask_in_level = (k == idx)
            rois_in_level = rois[mask_in_level]
            if self.method == 'pooling':
                pooled_feature = roi_pool(featmap_in_level, rois_in_level, output_size=self.output_size,
                                          spatial_scale=scale)
            elif self.method == 'align':
                pooled_feature = roi_align(featmap_in_level, rois_in_level, output_size=self.output_size,
                                           spatial_scale=scale, sampling_ratio=self.sampling_ratio)
            all_level_pooled_feature[mask_in_level] = pooled_feature

        return all_level_pooled_feature

    def _setup_scales(self, features, image_shapes):
        max_w = 0
        max_h = 0
        for shape in image_shapes:
            max_h = max(shape[0], max_h)
            max_w = max(shape[1], max_w)
        input_size = (max_h, max_w)

        scales = [self._get_scales(feat, input_size) for feat in features]
        return scales

    @staticmethod
    def _convert_to_roi_format(rois):
        rois = [
            torch.cat([torch.full((roi.shape[0], 1), batch_idx, device=roi.device), roi], dim=1)
            for batch_idx, roi in enumerate(rois)]
        rois = torch.cat(rois, dim=0)
        return rois

    @staticmethod
    def _filter_inputs(x, featmap_names):
        x_filtered = []
        for k, v in x.items():
            if k in featmap_names:
                x_filtered.append(v)
        return x_filtered

    @staticmethod
    def _get_scales(feature, image_size):
        size = feature.shape[-2:]
        scales = []
        for s1, s2 in zip(size, image_size):
            scale = float(s1) / float(s2)
            scale = 2 ** (float(torch.tensor(scale).log2().round()))
            scales.append(scale)
        return scales[0]


class MultiScaleRoIPooling(MultiScaleRoIOperation):
    def __init__(self, featmap_names, output_size):
        super(MultiScaleRoIPooling, self).__init__(featmap_names, output_size, method='pooling')

    def forward(self, features, rois, image_shape):
        """
        Arguments:
            features (Dict[Tensor]): FPN feature maps
            rois (List[Tensor[N, 4]]): proposal boxes
            image_shape (Tuple[H, W]): image shape
        Returns:
            Tensor:
                Pooled features
        """
        return super().forward(features, rois, image_shape)


class MultiScaleRoIAlign(MultiScaleRoIOperation):
    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__(featmap_names, output_size,
                                                 sampling_ratio=sampling_ratio, method='align')

    def forward(self, features, rois, image_shape):
        """
        Arguments:
            features (Dict[Tensor]): FPN feature maps
            rois (List[Tensor[N, 4]]): proposal boxes
            image_shape (Tuple[H, W]): image shape
        Returns:
            Tensor:
                aligned features
        """
        return super().forward(features, rois, image_shape)
