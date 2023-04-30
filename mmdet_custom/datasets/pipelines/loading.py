from mmdet.datasets.pipelines.loading import PIPELINES, LoadAnnotations


@PIPELINES.register_module()
class LoadKeypointAnnotations(LoadAnnotations):
    def _load_keypoints(self, results):
        anno_info = results['ann_info']
        results['gt_keypoints'] = anno_info['keypoints']
        results['keypoint_fields'].append('gt_keypoints')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        results = self._load_keypoints(results)
        return results

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += f'with_keypoint=True, '
        return repr_str
