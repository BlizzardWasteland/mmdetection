# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

def points_distance(points1,
                  points2,
                  mode='l2'):
    """Calculate the dis between each bbox of points1 and points2.

    Args:
        points1 (ndarray): Shape (n, 2)
        points2 (ndarray): Shape (k, 2)
        mode (str): l2 (l2 distance)

    Returns:
        distance (ndarray): Shape (n, k)
    """

    assert mode in ['l2']
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    rows = points1.shape[0]
    cols = points2.shape[0]
    dis = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return dis
    exchange = False
    if points1.shape[0] > points2.shape[0]:
        points1, points2 = points2, points1
        dis = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    for i in range(points1.shape[0]):
        dis_x = np.abs(points1[i, 0] - points2[:, 0]) 
        dis_y = np.abs(points1[i, 1] - points2[:, 1])
        dis[i, :] = dis_x + dis_y
    if exchange:
        dis = dis.T
    return dis


def _recalls(all_dis, proposal_nums, thrs):

    img_num = all_dis.shape[0]
    total_gt_num = sum([dis.shape[0] for dis in all_dis])

    _dis = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_dis = np.zeros(0)
        for i in range(img_num):
            dis = all_dis[i][:, :proposal_num].copy()
            gt_dis = np.zeros((dis.shape[0]))
            if dis.size == 0:
                tmp_dis = np.hstack((tmp_dis, gt_dis))
                continue
            for j in range(dis.shape[0]):
                gt_max_overlaps = dis.argmax(axis=1)
                max_dis = dis[np.arange(0, dis.shape[0]), gt_max_overlaps]
                gt_idx = max_dis.argmax()
                gt_dis[j] = max_dis[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                dis[gt_idx, :] = -1
                dis[:, box_idx] = -1
            tmp_dis = np.hstack((tmp_dis, gt_dis))
        _dis[k, :] = tmp_dis

    # _dis = np.fliplr(np.sort(_dis, axis=1))
    _dis = np.sort(_dis, axis=1)
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_dis <= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def set_recall_param(proposal_nums, dis_thrs):
    """Check proposal_nums and dis_thrs and set correct format."""
    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if dis_thrs is None:
        _dis_thrs = np.array([4])
    elif isinstance(dis_thrs, Sequence):
        _dis_thrs = np.array(dis_thrs)
    elif isinstance(dis_thrs, float):
        _dis_thrs = np.array([dis_thrs])
    else:
        _dis_thrs = dis_thrs

    return _proposal_nums, _dis_thrs


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 dis_thrs=4,
                 logger=None):
    """Calculate recalls.

    Args:
        gts (list[ndarray]): a list of arrays of shape (n, 2)
        proposals (list[ndarray]): a list of arrays of shape (k, 2) or (k, 3)
        proposal_nums (int | Sequence[int]): Top N proposals to be evaluated.
        dis_thrs (float | Sequence[float]): dis thresholds. Default: 4.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.

    Returns:
        ndarray: recalls of different dis and proposal nums
    """
    img_num = len(gts)
    assert img_num == len(proposals)
    proposal_nums, dis_thrs = set_recall_param(proposal_nums, dis_thrs)
    all_dis = []
    for i in range(img_num):
        proposals[i] = np.concatenate(proposals[i])
        print(proposals[i])
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 3:
            scores = proposals[i][:, 2]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            dis = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            dis = points_distance(
                gts[i],
                img_proposal[:prop_num, :2])
        all_dis.append(dis)
    all_dis = np.array(all_dis)
    recalls = _recalls(all_dis, proposal_nums, dis_thrs)

    print_recall_summary(recalls, proposal_nums, dis_thrs, logger=logger)
    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         dis_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    """Print recalls in a table.

    Args:
        recalls (ndarray): calculated from `bbox_recalls`
        proposal_nums (ndarray or list): top N proposals
        dis_thrs (ndarray or list): dis thresholds
        row_idxs (ndarray): which rows(proposal nums) to print
        col_idxs (ndarray): which cols(dis thresholds) to print
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    dis_thrs = np.array(dis_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(dis_thrs.size)
    row_header = [''] + dis_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)


def plot_num_recall(recalls, proposal_nums):
    """Plot Proposal_num-Recalls curve.

    Args:
        recalls(ndarray or list): shape (k,)
        proposal_nums(ndarray or list): same shape as `recalls`
    """
    if isinstance(proposal_nums, np.ndarray):
        _proposal_nums = proposal_nums.tolist()
    else:
        _proposal_nums = proposal_nums
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot([0] + _proposal_nums, [0] + _recalls)
    plt.xlabel('Proposal num')
    plt.ylabel('Recall')
    plt.axis([0, proposal_nums.max(), 0, 1])
    f.show()


def plot_dis_recall(recalls, dis_thrs):
    """Plot dis-Recalls curve.

    Args:
        recalls(ndarray or list): shape (k,)
        dis_thrs(ndarray or list): same shape as `recalls`
    """
    if isinstance(dis_thrs, np.ndarray):
        _dis_thrs = dis_thrs.tolist()
    else:
        _dis_thrs = dis_thrs
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot(_dis_thrs + [1.0], _recalls + [0.])
    plt.xlabel('dis')
    plt.ylabel('Recall')
    plt.axis([dis_thrs.min(), 1, 0, 1])
    f.show()
