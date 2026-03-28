import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class ClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = sorted(
            [name for name in os.listdir(root_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_default_num_workers():
    if os.name == "nt":
        return 1
    return min(4, os.cpu_count() or 1)


def describe_runtime(device, num_workers):
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({gpu_name}) | num_workers={num_workers}")
    else:
        print(f"Using device: cpu | num_workers={num_workers}")


def get_output_dir():
    return os.path.join("runs", "resnet50_5fold")


def save_training_plots(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=200)
    plt.close(fig)


def save_confusion_matrix(
    confusion_matrix,
    class_names,
    output_dir,
    filename="confusion_matrix.png",
    title="Confusion Matrix",
):
    fig, ax = plt.subplots(figsize=(20, 18))
    image = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(class_names))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def create_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def create_val_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_model(num_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    return model.to(device)


def mixup_batch(inputs, labels, alpha, device):
    if alpha <= 0.0:
        return inputs, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, labels, labels[index], lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)


def discover_class_names(root_dirs):
    class_names = None
    for root_dir in root_dirs:
        current = sorted(
            [
                name
                for name in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, name))
            ]
        )
        if class_names is None:
            class_names = current
        elif current != class_names:
            raise ValueError(f"Class folders do not match in {root_dir}.")
    return class_names


def collect_samples(root_dir, class_to_idx):
    samples = []
    for class_name, label in class_to_idx.items():
        class_dir = os.path.join(root_dir, class_name)
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                samples.append((os.path.join(class_dir, file_name), label))
    return samples


def build_stratified_folds(samples, num_folds, seed):
    targets = np.array([label for _, label in samples])
    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(num_folds)]

    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]
        rng.shuffle(class_indices)
        splits = np.array_split(class_indices, num_folds)
        for fold_idx, split in enumerate(splits):
            fold_indices[fold_idx].extend(split.tolist())

    return [sorted(indices) for indices in fold_indices]


def build_weighted_sampler(samples, num_classes):
    targets = torch.tensor([label for _, label in samples], dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets].double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate(model, data_loader, criterion, device, num_classes, collect_predictions=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            if collect_predictions:
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples

    confusion_matrix = None
    if collect_predictions:
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(confusion_matrix, (all_labels, all_predictions), 1)

    return avg_loss, avg_acc, confusion_matrix, all_labels, all_predictions


def predict_probabilities(model, test_loader, device, use_tta=True):
    model.eval()
    probabilities = []
    image_names = []

    with torch.no_grad():
        for inputs, names in tqdm(test_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            if use_tta:
                logits = (logits + model(torch.flip(inputs, dims=[3]))) / 2.0
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.cpu())
            image_names.extend(names)

    return torch.cat(probabilities, dim=0), image_names


def save_prediction(probabilities, image_names, class_names, output_dir):
    predicted_indices = probabilities.argmax(dim=1).tolist()
    records = []
    for image_name, predicted_idx in zip(image_names, predicted_indices):
        image_id, _ = os.path.splitext(image_name)
        records.append(
            {
                "image_name": image_id,
                "pred_label": int(class_names[predicted_idx]),
            }
        )

    df = pd.DataFrame(records)
    prediction_path = os.path.join(output_dir, "prediction.csv")
    df.to_csv(prediction_path, index=False)

    zip_path = os.path.join(output_dir, "prediction.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(prediction_path, arcname="prediction.csv")

    print(f"Prediction saved to {prediction_path}")
    print(f"Zip saved to {zip_path}")


def create_test_loader(test_dir, batch_size, num_workers, pin_memory):
    return DataLoader(
        TestDataset(test_dir, transform=create_val_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def predict_with_folds(output_dir, test_loader, device, class_names, num_classes, num_folds):
    ensemble_probabilities = None
    reference_names = None

    for fold_idx in range(num_folds):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        model_path = os.path.join(fold_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing fold checkpoint: {model_path}")

        model = build_model(num_classes, device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        fold_probabilities, image_names = predict_probabilities(
            model, test_loader, device, use_tta=True
        )

        if ensemble_probabilities is None:
            ensemble_probabilities = fold_probabilities
            reference_names = image_names
        else:
            if image_names != reference_names:
                raise ValueError("Test image order changed between folds.")
            ensemble_probabilities += fold_probabilities

        print(f"Fold {fold_idx + 1}/{num_folds} prediction complete.")

    ensemble_probabilities /= num_folds
    save_prediction(ensemble_probabilities, reference_names, class_names, output_dir)


def train_one_fold(
    fold_idx,
    num_folds,
    train_samples,
    val_samples,
    class_names,
    output_dir,
    batch_size,
    epochs,
    learning_rate,
    weight_decay,
    mixup_alpha,
    ema_decay,
    early_stop_patience,
    num_workers,
    device,
):
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    num_classes = len(class_names)
    pin_memory = torch.cuda.is_available()
    train_dataset = ClassificationDataset(train_samples, transform=create_train_transform())
    val_dataset = ClassificationDataset(val_samples, transform=create_val_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=build_weighted_sampler(train_samples, num_classes),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(num_classes, device)
    ema_model = ModelEMA(model, decay=ema_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total_train = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx + 1}/{num_folds} Epoch {epoch + 1}/{epochs}",
            leave=True,
        )
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_inputs, labels_a, labels_b, lam = mixup_batch(
                inputs, labels, mixup_alpha, device
            )

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_model.update(model)

            predictions = outputs.argmax(dim=1)
            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            running_correct += (
                lam * (predictions == labels_a).sum().item()
                + (1.0 - lam) * (predictions == labels_b).sum().item()
            )
            total_train += batch_size_now

            progress_bar.set_postfix(
                loss=f"{running_loss / total_train:.4f}",
                acc=f"{100.0 * running_correct / total_train:.2f}%",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        train_loss = running_loss / total_train
        train_acc = 100.0 * running_correct / total_train
        val_loss, val_acc, _, _, _ = evaluate(
            ema_model.module,
            val_loader,
            criterion,
            device,
            num_classes,
            collect_predictions=False,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Fold {fold_idx + 1} Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        improved_acc = val_acc > best_val_acc
        improved_loss = val_loss < best_val_loss - 1e-4

        if improved_acc or (abs(val_acc - best_val_acc) < 1e-8 and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(ema_model.module.state_dict(), os.path.join(fold_dir, "best_model.pth"))
            print(
                f"Fold {fold_idx + 1} saved new best model | "
                f"Val Acc: {best_val_acc:.2f}% | Val Loss: {best_val_loss:.4f}"
            )

        if improved_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print(f"Fold {fold_idx + 1} early stopping at epoch {epoch + 1}.")
            break

    with open(os.path.join(fold_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    save_training_plots(history, fold_dir)

    best_model = build_model(num_classes, device)
    best_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth"), map_location=device))
    val_loss, val_acc, confusion_matrix, labels, predictions = evaluate(
        best_model,
        val_loader,
        criterion,
        device,
        num_classes,
        collect_predictions=True,
    )
    save_confusion_matrix(
        confusion_matrix,
        class_names,
        fold_dir,
        title=f"Fold {fold_idx + 1} Confusion Matrix",
    )

    metrics = {
        "fold": fold_idx + 1,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "best_val_acc": val_acc,
        "best_val_loss": val_loss,
    }
    with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics, labels, predictions


def predict_only():
    output_dir = get_output_dir()
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    data_dir = "data"
    test_dir = os.path.join(data_dir, "test")
    batch_size = config["batch_size"]
    num_workers = 0 if os.name == "nt" else config["num_workers"]
    class_names = config["class_names"]
    num_classes = config["num_classes"]
    num_folds = config["num_folds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    describe_runtime(device, num_workers)

    test_loader = create_test_loader(test_dir, batch_size, num_workers, pin_memory)

    predict_with_folds(output_dir, test_loader, device, class_names, num_classes, num_folds)


def train():
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    output_dir = get_output_dir()
    batch_size = 64
    epochs = 35
    learning_rate = 1e-4
    weight_decay = 1e-2
    mixup_alpha = 0.2
    ema_decay = 0.999
    early_stop_patience = 8
    num_workers = get_default_num_workers()
    num_folds = 5
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    describe_runtime(device, num_workers)

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    class_names = discover_class_names([train_dir, val_dir])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    all_samples = collect_samples(train_dir, class_to_idx) + collect_samples(val_dir, class_to_idx)
    fold_val_indices = build_stratified_folds(all_samples, num_folds, seed)

    config = {
        "model_name": "resnet50_5fold",
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "mixup_alpha": mixup_alpha,
        "ema_decay": ema_decay,
        "early_stop_patience": early_stop_patience,
        "num_workers": num_workers,
        "num_folds": num_folds,
        "seed": seed,
        "device": str(device),
        "num_classes": len(class_names),
        "class_names": class_names,
        "num_labeled_samples": len(all_samples),
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    fold_metrics = []
    oof_labels = []
    oof_predictions = []

    for fold_idx, val_indices in enumerate(fold_val_indices):
        val_index_set = set(val_indices)
        train_samples = [sample for idx, sample in enumerate(all_samples) if idx not in val_index_set]
        val_samples = [all_samples[idx] for idx in val_indices]

        metrics, labels, predictions = train_one_fold(
            fold_idx=fold_idx,
            num_folds=num_folds,
            train_samples=train_samples,
            val_samples=val_samples,
            class_names=class_names,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mixup_alpha=mixup_alpha,
            ema_decay=ema_decay,
            early_stop_patience=early_stop_patience,
            num_workers=num_workers,
            device=device,
        )
        fold_metrics.append(metrics)
        oof_labels.extend(labels)
        oof_predictions.extend(predictions)

    oof_confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(oof_confusion_matrix, (oof_labels, oof_predictions), 1)
    oof_accuracy = 100.0 * np.mean(np.array(oof_labels) == np.array(oof_predictions))
    save_confusion_matrix(
        oof_confusion_matrix,
        class_names,
        output_dir,
        filename="oof_confusion_matrix.png",
        title="5-Fold OOF Confusion Matrix",
    )

    summary = {
        "fold_metrics": fold_metrics,
        "mean_val_acc": float(np.mean([metric["best_val_acc"] for metric in fold_metrics])),
        "std_val_acc": float(np.std([metric["best_val_acc"] for metric in fold_metrics])),
        "mean_val_loss": float(np.mean([metric["best_val_loss"] for metric in fold_metrics])),
        "oof_acc": float(oof_accuracy),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    test_loader = create_test_loader(test_dir, batch_size, num_workers, pin_memory)
    predict_with_folds(
        output_dir=output_dir,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        num_classes=len(class_names),
        num_folds=num_folds,
    )

    print(
        f"5-fold training complete | OOF Acc: {oof_accuracy:.2f}% | "
        f"Mean Fold Val Acc: {summary['mean_val_acc']:.2f}%"
    )


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training",
    )
    args = parser.parse_args()

    if args.predict_only:
        predict_only()
    else:
        train()
