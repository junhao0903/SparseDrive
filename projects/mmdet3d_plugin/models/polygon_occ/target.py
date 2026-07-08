import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS

from ..map.target import SparsePoint3DTarget


@BBOX_SAMPLERS.register_module()
class SparsePolygonTarget(SparsePoint3DTarget):
    def sample(
        self,
        cls_preds,
        pts_preds,
        cls_targets,
        pts_targets,
    ):
        pts_targets = [x.flatten(2, 3) if len(x.shape) == 4 else x for x in pts_targets]
        indices = []
        for cls_pred, pts_pred, cls_target, pts_target in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds = dict(lines=pts_pred, scores=cls_pred)
            gts = dict(lines=pts_target, labels=cls_target)
            indices.append(self.assigner.assign(preds, gts))

        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones(
            [bs, num_pred], dtype=torch.long
        ) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            if gt_permute_index is None:
                output_box_target[i, pred_idx] = pts_targets[i][target_idx]
            else:
                permute_idx = gt_permute_index[pred_idx, target_idx]
                output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            output_reg_weights[i, pred_idx] = 1

        return output_cls_target, output_box_target, output_reg_weights
