import torch


def focal_loss(pred, gt, alpha=4, beta=4):
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()

    # clamp to avoid log(0)
    pred = pred.clamp(1e-6, 1 - 1e-6)

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = (
        -((1 - gt) ** beta)
        * (pred ** alpha)
        * torch.log(1 - pred)
        * neg_mask
    )

    num_pos = pos_mask.sum()
    if num_pos == 0:
        return neg_loss.sum()

    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def reg_l1_loss(pred, gt, mask):
    loss = torch.abs(pred - gt) * mask
    num_pos = mask.sum() + 1e-6
    return loss.sum() / num_pos


def centernet_loss(pred, gt, offset_weight=1.0, size_weight=1.0):
    heatmap, offset, size = pred
    gt_heatmap, gt_offset, gt_size, reg_mask = gt

    # Soft mask: blend predicted confidence with GT to keep FNs supervised
    # - At GT centers: max(pred, 1.0) = 1.0 → always full weight (covers FNs)
    # - Near GT centers: Gaussian tail provides baseline weight
    # - pred_heatmap adds weight where network is confident
    soft_weight = torch.max(heatmap.detach(), gt_heatmap)
    soft_mask = soft_weight.unsqueeze(1).expand_as(reg_mask) * reg_mask

    l_heatmap = focal_loss(heatmap, gt_heatmap)
    l_offset = reg_l1_loss(offset, gt_offset, soft_mask)
    l_size = reg_l1_loss(size, gt_size, soft_mask)

    loss = l_heatmap + offset_weight * l_offset + size_weight * l_size
    return loss, l_heatmap, l_offset, l_size
