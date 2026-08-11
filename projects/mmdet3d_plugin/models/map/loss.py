import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss


@LOSSES.register_module()
class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight


@LOSSES.register_module()
class PolygonChamferLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if pred.numel() == 0:
            return pred.sum()

        dists = torch.cdist(pred, target, p=1)
        pred_to_target = dists.min(dim=-1).values
        target_to_pred = dists.min(dim=-2).values
        loss = 0.5 * (pred_to_target.mean(dim=-1) + target_to_pred.mean(dim=-1))

        if weight is not None:
            sample_weight = weight.view(weight.shape[0], -1).mean(dim=-1)
            loss = loss * sample_weight

        if reduction == 'none':
            return loss * self.loss_weight
        if reduction == 'sum':
            loss = loss.sum()
        else:
            if avg_factor is not None:
                loss = loss.sum() / max(float(avg_factor), 1.0)
            else:
                loss = loss.mean()
        return loss * self.loss_weight


@LOSSES.register_module()
class PolygonAreaLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if pred.numel() == 0:
            return pred.sum()

        pred_area = self.compute_area(pred)
        target_area = self.compute_area(target)
        loss = (pred_area - target_area).abs()

        if weight is not None:
            sample_weight = weight.view(weight.shape[0], -1).mean(dim=-1)
            loss = loss * sample_weight

        if reduction == 'none':
            return loss * self.loss_weight
        if reduction == 'sum':
            loss = loss.sum()
        else:
            if avg_factor is not None:
                loss = loss.sum() / max(float(avg_factor), 1.0)
            else:
                loss = loss.mean()
        return loss * self.loss_weight

    def compute_area(self, polygon):
        x = polygon[..., 0]
        y = polygon[..., 1]
        x_next = torch.roll(x, shifts=-1, dims=-1)
        y_next = torch.roll(y, shifts=-1, dims=-1)
        return 0.5 * (x * y_next - y * x_next).sum(dim=-1).abs()


@LOSSES.register_module()
class PolygonRasterLoss(nn.Module):
    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        grid_size=(32, 64),
        sharpness=10.0,
        eps=1e-6,
        bce_weight=0.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.grid_size = grid_size
        self.sharpness = sharpness
        self.eps = eps
        self.bce_weight = bce_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if pred.numel() == 0:
            return pred.sum()

        pred_mask = self.soft_rasterize(pred)
        target_mask = self.soft_rasterize(target).detach()

        pred_flat = pred_mask.flatten(1)
        target_flat = target_mask.flatten(1)
        intersection = (pred_flat * target_flat).sum(dim=-1)
        denom = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)
        loss = dice_loss

        if self.bce_weight > 0:
            bce = nn.functional.binary_cross_entropy(
                pred_mask.clamp(self.eps, 1.0 - self.eps),
                target_mask,
                reduction='none',
            ).flatten(1).mean(dim=-1)
            loss = loss + self.bce_weight * bce

        if weight is not None:
            sample_weight = weight.view(weight.shape[0], -1).mean(dim=-1)
            loss = loss * sample_weight

        if reduction == 'none':
            return loss * self.loss_weight
        if reduction == 'sum':
            loss = loss.sum()
        else:
            if avg_factor is not None:
                loss = loss.sum() / max(float(avg_factor), 1.0)
            else:
                loss = loss.mean()
        return loss * self.loss_weight

    def soft_rasterize(self, polygon):
        grid = self.build_grid(polygon)
        v0 = polygon.unsqueeze(1).unsqueeze(1)
        v1 = torch.roll(polygon, shifts=-1, dims=1).unsqueeze(1).unsqueeze(1)
        p = grid.unsqueeze(-2)

        rel0 = v0 - p
        rel1 = v1 - p
        cross = rel0[..., 0] * rel1[..., 1] - rel0[..., 1] * rel1[..., 0]
        dot = (rel0 * rel1).sum(dim=-1)
        angles = torch.atan2(cross, dot + self.eps)
        winding = angles.sum(dim=-1).abs()
        return torch.sigmoid((winding - torch.pi) * self.sharpness)

    def build_grid(self, polygon):
        h, w = self.grid_size
        ys = torch.linspace(
            0.5 / h,
            1.0 - 0.5 / h,
            h,
            device=polygon.device,
            dtype=polygon.dtype,
        )
        xs = torch.linspace(
            0.5 / w,
            1.0 - 0.5 / w,
            w,
            device=polygon.device,
            dtype=polygon.dtype,
        )
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx, yy], dim=-1)


@LOSSES.register_module()
class SparseLineLoss(nn.Module):
    def __init__(
        self,
        loss_line,
        loss_chamfer=None,
        loss_area=None,
        loss_raster=None,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_line = build(loss_line, LOSSES)
        self.loss_chamfer = build(loss_chamfer, LOSSES)
        self.loss_area = build(loss_area, LOSSES)
        self.loss_raster = build(loss_raster, LOSSES)
        self.num_sample = num_sample
        self.roi_size = roi_size

    def forward(
        self,
        line,
        line_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):

        output = {}
        line = self.reshape_line(line)
        line_target = self.reshape_line(line_target)
        line_norm = self.normalize_line(line)
        line_target_norm = self.normalize_line(line_target)
        line_norm_flat = self.flatten_line(line_norm)
        line_target_norm_flat = self.flatten_line(line_target_norm)
        flat_weight = self.flatten_weight(weight)
        line_loss = self.loss_line(
            line_norm_flat,
            line_target_norm_flat,
            weight=weight,
            avg_factor=avg_factor,
        )
        output[f"{prefix}loss_line{suffix}"] = line_loss

        if self.loss_chamfer is not None:
            chamfer_loss = self.loss_chamfer(
                line_norm,
                line_target_norm,
                weight=flat_weight,
                avg_factor=avg_factor,
            )
            output[f"{prefix}loss_chamfer{suffix}"] = chamfer_loss

        if self.loss_area is not None:
            area_loss = self.loss_area(
                line_norm,
                line_target_norm,
                weight=flat_weight,
                avg_factor=avg_factor,
            )
            output[f"{prefix}loss_area{suffix}"] = area_loss

        if self.loss_raster is not None:
            raster_loss = self.loss_raster(
                line_norm,
                line_target_norm,
                weight=flat_weight,
                avg_factor=avg_factor,
            )
            output[f"{prefix}loss_raster{suffix}"] = raster_loss

        return output

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line

        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm

        return line

    def reshape_line(self, line):
        coords_dim = max(line.shape[-1] // max(self.num_sample, 1), 1)
        if line.shape[0] == 0:
            return line.view(line.shape[:-1] + (self.num_sample, coords_dim))
        line = line.view(line.shape[:-1] + (self.num_sample, coords_dim))
        return line

    def flatten_weight(self, weight):
        if weight is None:
            return None
        if weight.shape[0] == 0:
            return weight.new_zeros((0, self.num_sample))
        weight = weight.view(weight.shape[:-1] + (self.num_sample, -1))
        return weight.mean(dim=-1)

    def normalize_line_flat(self, line):
        line = self.normalize_line(line)
        return self.flatten_line(line)

    def flatten_line(self, line):
        if line.shape[0] == 0:
            return line.flatten(-2, -1)
        return line.flatten(-2, -1)
