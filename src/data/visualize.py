import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from data.centernet_dataset import CenterNetDataset


def visualize_sample(dataset, index=0):
    sample = dataset[index]

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # Image (C, H, W) -> (H, W, C)
    axes[0, 0].imshow(sample["img"].permute(1, 2, 0))
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    # Segmentation — distinct color per agent, background black
    seg = sample["gt_seg"].numpy()
    ids = np.unique(seg)
    ids = ids[ids != 0]  # exclude background
    n = len(ids)
    colors = ["black"] + [plt.cm.tab20(i / max(n, 1)) for i in range(n)]
    id_map = {0: 0}
    for i, aid in enumerate(ids):
        id_map[aid] = i + 1
    mapped = np.vectorize(id_map.get)(seg)
    cmap = mcolors.ListedColormap(colors)
    axes[0, 1].imshow(mapped, cmap=cmap, interpolation="nearest", vmin=0, vmax=n)
    axes[0, 1].set_title(f"Segmentation GT ({n} agents)")
    axes[0, 1].axis("off")

    # Heatmap
    axes[0, 2].imshow(sample["gt_heat"], cmap="hot")
    axes[0, 2].set_title("Heatmap GT")
    axes[0, 2].axis("off")

    # Reg mask — show where offset/size are supervised
    axes[1, 0].imshow(sample["reg_mask"][0], cmap="gray")
    axes[1, 0].set_title(f"Reg Mask ({int(sample["reg_mask"][0].sum())} centers)")
    axes[1, 0].axis("off")

    # Offset — visualize x and y offset as a 2-channel color image
    offset = sample["gt_offset"].numpy()
    offset_vis = np.zeros((*offset.shape[1:], 3))
    offset_vis[:, :, 0] = offset[0]  # x offset -> red
    offset_vis[:, :, 1] = offset[1]  # y offset -> green
    axes[1, 1].imshow(offset_vis)
    axes[1, 1].set_title("Offset GT (R=dx, G=dy)")
    axes[1, 1].axis("off")

    # Size — visualize w and h as a 2-channel color image
    size = sample["gt_size"].numpy()
    size_vis = np.zeros((*size.shape[1:], 3))
    if size.max() > 0:
        size_vis[:, :, 0] = size[0] / size.max()  # width -> red
        size_vis[:, :, 1] = size[1] / size.max()  # height -> green
    axes[1, 2].imshow(size_vis)
    axes[1, 2].set_title("Size GT (R=w, G=h)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("visualization.png", dpi=150)
    print("Saved to visualization.png")


def visualize_prediction(model, dataset, epoch, index=0):
    sample = dataset[index]
    img = sample["img"].unsqueeze(0)

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        heatmap, offset, size = model(img.to(device))
        heatmap = heatmap[0].cpu()
        offset = offset[0].cpu()
        size = size[0].cpu()
    model.train()

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # Row 1: GT
    axes[0, 0].imshow(sample["img"].permute(1, 2, 0))
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sample["gt_heat"], cmap="hot")
    axes[0, 1].set_title("Heatmap GT")
    axes[0, 1].axis("off")

    gt_size = sample["gt_size"].numpy()
    gt_size_vis = np.zeros((*gt_size.shape[1:], 3))
    if gt_size.max() > 0:
        gt_size_vis[:, :, 0] = gt_size[0] / gt_size.max()
        gt_size_vis[:, :, 1] = gt_size[1] / gt_size.max()
    axes[0, 2].imshow(gt_size_vis)
    axes[0, 2].set_title("Size GT (R=w, G=h)")
    axes[0, 2].axis("off")

    # Row 2: Predictions
    axes[1, 0].imshow(heatmap, cmap="hot")
    axes[1, 0].set_title("Heatmap Pred")
    axes[1, 0].axis("off")

    off_np = offset.numpy()
    off_vis = np.zeros((*off_np.shape[1:], 3))
    off_vis[:, :, 0] = np.clip(off_np[0], 0, 1)
    off_vis[:, :, 1] = np.clip(off_np[1], 0, 1)
    axes[1, 1].imshow(off_vis)
    axes[1, 1].set_title("Offset Pred (R=dx, G=dy)")
    axes[1, 1].axis("off")

    size_np = size.numpy()
    size_vis = np.zeros((*size_np.shape[1:], 3))
    if size_np.max() > 0:
        size_vis[:, :, 0] = size_np[0] / size_np.max()
        size_vis[:, :, 1] = size_np[1] / size_np.max()
    axes[1, 2].imshow(np.clip(size_vis, 0, 1))
    axes[1, 2].set_title("Size Pred (R=w, G=h)")
    axes[1, 2].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=20)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    dataset = CenterNetDataset("../prepared_dataset")
    visualize_sample(dataset, index=0)