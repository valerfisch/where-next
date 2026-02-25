import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from data.centernet_dataset import CenterNetDataset


def visualize_sample(dataset, index=0):
    sample = dataset[index]

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Image (C, H, W) -> (H, W, C)
    axes[0].imshow(sample.img.permute(1, 2, 0))
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Segmentation — distinct color per agent, background black
    seg = sample.gt_seg.numpy()
    ids = np.unique(seg)
    ids = ids[ids != 0]  # exclude background
    n = len(ids)
    colors = ["black"] + [plt.cm.tab20(i / max(n, 1)) for i in range(n)]
    id_map = {0: 0}
    for i, aid in enumerate(ids):
        id_map[aid] = i + 1
    mapped = np.vectorize(id_map.get)(seg)
    cmap = mcolors.ListedColormap(colors)
    axes[1].imshow(mapped, cmap=cmap, interpolation="nearest", vmin=0, vmax=n)
    axes[1].set_title(f"Segmentation GT ({n} agents)")
    axes[1].axis("off")

    # Heatmap
    axes[2].imshow(sample.gt_heat, cmap="hot")
    axes[2].set_title("Heatmap GT")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("visualization.png", dpi=150)
    print("Saved to visualization.png")


if __name__ == "__main__":
    dataset = CenterNetDataset("../prepared_dataset")
    visualize_sample(dataset, index=0)