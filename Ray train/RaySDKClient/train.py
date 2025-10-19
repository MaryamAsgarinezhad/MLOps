#!/usr/bin/env python

import os
from typing import Dict

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from torchvision.models import VisionTransformer
from tqdm import tqdm
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import tempfile
import uuid
import ray


def get_dataloaders(batch_size):
    # Transform to normalize the input images
    transform = transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    with FileLock(os.path.expanduser("~/data.lock")):
        # Download data from open datasets.
        training_data = datasets.CIFAR10(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )
        
        testing_data = datasets.CIFAR10(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    train_dataloader, valid_dataloader = get_dataloaders(batch_size=batch_size)

    # [1] Prepare data loader for distributed training.
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    valid_dataloader = ray.train.torch.prepare_data_loader(valid_dataloader)

    model = VisionTransformer(
        image_size=32,   # CIFAR-10 image size is 32x32
        patch_size=4,    # Patch size is 4x4
        num_layers=12,   # Number of transformer layers
        num_heads=8,     # Number of attention heads
        hidden_dim=384,  # Hidden size (can be adjusted)
        mlp_dim=768,     # MLP dimension (can be adjusted)
        num_classes=10   # CIFAR-10 has 10 classes
    )

    # [2] Prepare and wrap your model with DistributedDataParallel.
    # =======================================================================
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # training loop.
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)  # Required for the distributed sampler to shuffle properly across epochs.

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(valid_dataloader, desc=f"Valid Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)

                valid_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        valid_loss /= len(train_dataloader)
        accuracy = num_correct / num_total


        # [3] (Optional) Report checkpoints and attached metrics to Ray Train.
        # ====================================================================
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics={"loss": valid_loss, "accuracy": accuracy},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
            if ray.train.get_context().get_world_rank() == 0:
                print({"epoch_num": epoch, "loss": valid_loss, "accuracy": accuracy})


def train_cifar_10(num_workers, use_gpu):
    runtime_env = {
        "pip": [
            "torch>=2.0",
            "torchvision>=0.15", 
            "ray[train]>=2.48",
            "filelock",
            "tqdm"
        ],
    }
        
    global_batch_size = 512

    train_config = {
        "lr": 1e-3,
        "epochs": 1,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # =============================================
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    run_config = RunConfig(
        # /mnt/cluster_storage is an Anyscale-specific storage path.
        storage_path="/mnt/cluster_storage", 
        name=f"train_run-{uuid.uuid4().hex}",
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        # run_config=run_config,
        runtime_env=runtime_env,
    )
    result = trainer.fit()
    print(f"Training result: {result}")

if __name__ == "__main__":
    train_cifar_10(num_workers=2, use_gpu=True)

# import torch
# from ray.air import ScalingConfig
# from ray.train.torch import TorchTrainer  # Use TorchTrainer for PyTorch
# from ray.air.config import ScalingConfig
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from sklearn.datasets import fetch_20newsgroups
# from torch.utils.data import Dataset, DataLoader

# # port-forward ray service before running the code
# # ray.init(address="127.0.0.1:6379")  # when not using job submit

# class NewsgroupsDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=128,
#         )
#         return {key: value.squeeze(0) for key, value in inputs.items()}, torch.tensor(label)

# def train_loop_per_worker(config):
#     model_name = "distilbert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     train_dataset = NewsgroupsDataset(config["texts"], config["labels"], tokenizer)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         num_workers=4,
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

#     for epoch in range(config["epochs"]):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             inputs, labels = batch
#             inputs = {key: value.to(device) for key, value in inputs.items()}
#             labels = labels.to(device)

#             outputs = model(**inputs, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss}")

#     return total_loss

# newsgroups_train = fetch_20newsgroups(subset='train')
# X_train, y_train = newsgroups_train.data, newsgroups_train.target

# config = {
#     "epochs": 3,
#     "batch_size": 16,
#     "lr": 2e-5,
#     "texts": X_train,
#     "labels": y_train
# }

# trainer = TorchTrainer(
#     train_loop_per_worker=train_loop_per_worker,
#     scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
#     train_loop_config=config
# )

# result = trainer.fit()
# print(f"Final training result: {result.metrics}")
