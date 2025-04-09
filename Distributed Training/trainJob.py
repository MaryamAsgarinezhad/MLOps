from ray.air import ScalingConfig
from ray.train.torch import TorchTrainer  # Use TorchTrainer for PyTorch
from ray.air.config import ScalingConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset, DataLoader
import torch

# port-forward ray service before running the code
# ray.init(address="127.0.0.1:6379")  # when not using job submit

class NewsgroupsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        return {key: value.squeeze(0) for key, value in inputs.items()}, torch.tensor(label)

def train_loop_per_worker(config):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = NewsgroupsDataset(config["texts"], config["labels"], tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss}")

    return total_loss

newsgroups_train = fetch_20newsgroups(subset='train')
X_train, y_train = newsgroups_train.data, newsgroups_train.target

config = {
    "epochs": 3,
    "batch_size": 16,
    "lr": 2e-5,
    "texts": X_train,
    "labels": y_train
}

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    train_loop_config=config
)

result = trainer.fit()
print(f"Final training result: {result.metrics}")

