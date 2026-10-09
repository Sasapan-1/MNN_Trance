from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_core import (
    DeepBMNN_11_layer,
    DeepBMNN_6_layer_reverse_binary,
)


MODELS = {
    "bmnn_6": (DeepBMNN_6_layer_reverse_binary, "MNIST_DeepBMNNmodel1v5_6_r_para.pth"),
    "deep_bmnn_11": (DeepBMNN_11_layer, "MNIST_DeepBMNNmodel1v5_11_para.pth"),
}


def make_loader(data_dir: Path, train: bool, batch_size: int, limit: int | None) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss_sum += F.nll_loss(output, labels, reduction="sum").item()
        correct += int((output.argmax(dim=1) == labels).sum().item())
    return correct / len(loader.dataset), loss_sum / len(loader.dataset)


def train(args: argparse.Namespace) -> Path:
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model_class, filename = MODELS[args.model]
    model = model_class().to(args.device)
    optimizer = torch_optimizer.RAdam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    train_loader = make_loader(args.data_dir, True, args.batch_size, args.max_train)
    test_loader = make_loader(args.data_dir, False, args.batch_size, args.max_test)

    history = {"train_A": [], "train_L": [], "test_A": [], "test_L": []}
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["MNIST_DeepBMNNmodel1v5"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        for key in history:
            history[key] = list(checkpoint.get(key, []))

    if args.epochs <= start_epoch:
        raise ValueError(f"--epochs must exceed resumed epoch {start_epoch}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / filename
    if checkpoint_path.exists() and args.resume is None:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint_path}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(images), labels)
            loss.backward()
            optimizer.step()

        train_accuracy, train_loss = evaluate(model, train_loader, args.device)
        test_accuracy, test_loss = evaluate(model, test_loader, args.device)
        history["train_A"].append(train_accuracy)
        history["train_L"].append(train_loss)
        history["test_A"].append(test_accuracy)
        history["test_L"].append(test_loss)
        torch.save(
            {
                "MNIST_DeepBMNNmodel1v5": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                **history,
            },
            checkpoint_path,
        )
        print(
            f"epoch={epoch} train_accuracy={train_accuracy:.4f} "
            f"test_accuracy={test_accuracy:.4f} checkpoint={checkpoint_path}",
            flush=True,
        )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the collected 6- or 11-layer MNIST model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=700, help="Total target epoch count.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "training_runs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
