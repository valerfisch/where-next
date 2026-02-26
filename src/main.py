import math
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.centernet_dataset import CenterNetDataset
from data.visualize import visualize_prediction
from model.centernet import CenterNet
from losses import centernet_loss

# TODO: move to config file
max_lr = 5e-4
min_lr = 1e-6
warmup_epochs = 5
epochs = 10000

# TODO: Cleanup file into multiple ones, more abstraction, less distraction
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Start loading the Dataset")
    dataset = CenterNetDataset("../prepared_dataset")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print("Dataset loaded")

    model = CenterNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    writer = SummaryWriter("runs/centernet")

    global_step = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_start = time.time()

        # Learning rate schedule: linear warmup then cosine decay
        if epoch < warmup_epochs:
            lr = min_lr + (max_lr - min_lr) * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if epoch == 0 or (epoch + 1) % 10 == 0:
            fig = visualize_prediction(model, dataset, epoch)
            writer.add_figure("predictions", fig, epoch)

        epoch_loss = 0.0
        epoch_heat = 0.0
        epoch_off = 0.0
        epoch_size = 0.0

        for i, batch in enumerate(dataloader):
            img = batch["img"].to(device)
            gt_heat = batch["gt_heat"].to(device)
            gt_offset = batch["gt_offset"].to(device)
            gt_size = batch["gt_size"].to(device)
            reg_mask = batch["reg_mask"].to(device)

            optimizer.zero_grad()

            pred = model(img)
            gt = (gt_heat, gt_offset, gt_size, reg_mask)
            loss, l_heat, l_off, l_size = centernet_loss(pred, gt)

            loss.backward()
            optimizer.step()

            writer.add_scalar("step/loss", loss.item(), global_step)
            writer.add_scalar("step/loss_heatmap", l_heat.item(), global_step)
            writer.add_scalar("step/loss_offset", l_off.item(), global_step)
            writer.add_scalar("step/loss_size", l_size.item(), global_step)
            global_step += 1

            epoch_loss += loss.item()
            epoch_heat += l_heat.item()
            epoch_off += l_off.item()
            epoch_size += l_size.item()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f} (heat: {l_heat.item():.4f}, "
                    f"off: {l_off.item():.4f}, size: {l_size.item():.4f})"
                )

        n_batches = len(dataloader)
        writer.add_scalar("epoch/loss", epoch_loss / n_batches, epoch)
        writer.add_scalar("epoch/loss_heatmap", epoch_heat / n_batches, epoch)
        writer.add_scalar("epoch/loss_offset", epoch_off / n_batches, epoch)
        writer.add_scalar("epoch/loss_size", epoch_size / n_batches, epoch)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("epoch/time", time.time() - epoch_start, epoch)

    fig = visualize_prediction(model, dataset, epochs)
    writer.add_figure("predictions", fig, epochs)
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
