import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import *

@PIPELINES.register_module()
class KPad(Pad):
    pass

@PIPELINES.register_module()
class KResize(Resize):
    def _resize_keypoints(self, results):
        for key in results.get('keypoint_fields', []):
            keypoints = results[key].copy()
            n_sets = keypoints.shape[0]
            for i in range(n_sets):
                keypoints[i, :, :2] = keypoints[
                    i, :, :2] * results['scale_factor'][:2]
            results[key] = keypoints
    
    def __call__(self, results):
        results = super().__call__(results)
        self._resize_keypoints(results)
        return results

@PIPELINES.register_module()
class KRandomFlip(RandomFlip):
    def keypoints_flip(self, keypoints, img_shape, direction):
        """Flip keypoints horizontally or vertically.
        Args:
            keypoints(ndarray): shape (..., 2)
            img_shape(tuple): (height, width)
        """
        # assert keypoints.shape[-1] == 3
        assert keypoints.shape[-1] == 2
        flipped = keypoints.copy()
        h = img_shape[0]
        w = img_shape[1]
        if direction == 'horizontal':
            flipped[..., 0] = w - keypoints[..., 0]
        elif direction == 'vertical':
            flipped[..., 1] = h - keypoints[..., 1]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        for kpts_set in flipped:
            invisible = np.logical_or(
                np.any(kpts_set[:, :2] < 0, axis=1),
                np.logical_or(kpts_set[:, 0] >= h, kpts_set[:, 1] >= w))
            # kpts_set[invisible, 2] = 0
        return flipped
    
    def __call__(self, results):
        results = super().__call__(results)
        if results['flip']:
            # flip keypoints
            for key in results.get('keypoint_fields', []):
                results[key] = self.keypoints_flip(results[key],
                                                   results['img_shape'],
                                                   results['flip_direction'])
        return results
    
@PIPELINES.register_module()
class KRandomCrop(RandomCrop):
    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # crop keypoints accordingly and clip to the image boundary
        for key in results.get('keypoint_fields', []):
            keypoint_offset = np.array([offset_w, offset_h, 0])
            keypoints = results[key] - keypoint_offset
            outside_point_idx = (keypoints[..., 0] < 0) + \
                                (keypoints[..., 0] > (img.shape[1] - 1)) + \
                                (keypoints[..., 1] < 0) + \
                                (keypoints[..., 1] > (img.shape[0] - 1))
            keypoints[outside_point_idx, 2] = 0
            if np.all(outside_point_idx):
                return None  # No valid keypoint found
            results[key] = keypoints
        return results

@PIPELINES.register_module()
class KExpand(Expand):
    def __call__(self, results):
        """Call function to expand images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)

        # expand masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segs
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results[key] = expand_gt_seg

        # expand keypoints
        for key in results.get('keypoint_fields', []):
            results[key] = results[key] + np.array([left, top]).astype(results[key].dtype)
        return results

@PIPELINES.register_module()
class KMinIoURandomCrop(MinIoURandomCrop):
    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        if self.bbox_clip_border:
                            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]
                # expand keypoints
                for key in results.get('keypoint_fields', []):
                    offset = np.array([patch[0], patch[1]]).astype(boxes.dtype)
                    keypoints = results[key] - offset
                    outside_point_idx = (keypoints[..., 0] < 0) + \
                        (keypoints[..., 0] > (patch[2] - patch[0] - 1)) + \
                        (keypoints[..., 1] < 0) + \
                        (keypoints[..., 1] > (patch[3] - patch[1] - 1))
                    # keypoints[outside_point_idx, 2] = 0
                    if np.all(outside_point_idx):
                        return None  # No valid keypoint found
                    results[key] = keypoints

                return results

@PIPELINES.register_module()
class KAlbu(Albu):
    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes',
                'gt_keypoints': 'keypoints'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}
        
    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: add keypoints_fields
        if 'keypoints' in results:
            n_kpts_per_class = results['keypoints'].shape[0]
            converted_kpts = []
            cols, rows, _ = results['img_shape']
            kpts_vis = results[
                'keypoints'][:, :,
                             2]  # Keep a copy of keypoints visibility flag
            for x in range(n_kpts_per_class):
                kpts = results['keypoints'][x, :, :2]
                new_kpts = keypoints_utils.convert_keypoints_to_albumentations(
                    kpts, 'xy', rows, cols)
                converted_kpts.extend(new_kpts)
            results['keypoints'] = converted_kpts

        # TODO: Support mask structure in albu
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            results['masks'] = results['masks'].masks

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        if 'keypoints' in results:
            cols, rows, _ = results['img_shape']
            kpts = results['keypoints']
            new_kpts = keypoints_utils.convert_keypoints_from_albumentations(
                kpts, 'xy', rows, cols)
            new_kpts = np.array(new_kpts).reshape(n_kpts_per_class, -1, 2)
            pts = np.zeros((n_kpts_per_class, new_kpts.shape[1], 3))
            pts[:, :, :2] = new_kpts
            pts[:, :, 2] = kpts_vis

            outside_point_idx = (pts[..., 0] < 0) + \
                                (pts[..., 0] > (cols - 1)) + \
                                (pts[..., 1] < 0) + \
                                (pts[..., 1] > (rows - 1))
            pts[outside_point_idx, 2] = 0
            if np.all(outside_point_idx) and self.skip_img_without_anno:
                return None  # No valid keypoint found
            results['keypoints'] = pts

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results


@PIPELINES.register_module()
class KRandomCenterCropPad(RandomCenterCropPad):
    def _train_aug(self, results):
        """Random crop and around padding the original image.
        Args:
            results (dict): Image infomations in the augment pipeline.
        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        boxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                mask = self._filter_boxes(patch, boxes)
                # if image do not have valid bbox, any crop patch is valid.
                if not mask.any() and len(boxes) > 0:
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape
                results['pad_shape'] = cropped_img.shape

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                for key in results.get('bbox_fields', []):
                    mask = self._filter_boxes(patch, results[key])
                    bboxes = results[key][mask]
                    bboxes[:, 0:4:2] += cropped_center_x - left_w - x0
                    bboxes[:, 1:4:2] += cropped_center_y - top_h - y0
                    bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
                    bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
                    keep = (bboxes[:, 2] > bboxes[:, 0]) & (
                        bboxes[:, 3] > bboxes[:, 1])
                    bboxes = bboxes[keep]
                    results[key] = bboxes
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            labels = results['gt_labels'][mask]
                            labels = labels[keep]
                            results['gt_labels'] = labels
                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomCenterCropPad only supports bbox.')

                # crop keypoints accordingly and clip to the image boundary
                for key in results.get('keypoint_fields', []):
                    kpts = results[key]
                    kpts[:, :, 0] += cropped_center_x - left_w - x0
                    kpts[:, :, 1] += cropped_center_y - top_h - y0
                    keep = (kpts[:, :, 0] >= 0) & (kpts[:, :, 0] < new_w) & (
                        kpts[:, :, 1] >= 0) & (
                            kpts[:, :, 1] < new_h)
                    kpts[np.invert(keep), 2] = 0
                    results[key] = kpts

                # crop semantic seg
                for key in results.get('seg_fields', []):
                    raise NotImplementedError(
                        'RandomCenterCropPad only supports bbox.')
                return results

@PIPELINES.register_module()
class RangeClip:
    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, results):
        results['img'] = results['img'].clip(
            self.min_val, self.max_val)
        return results
