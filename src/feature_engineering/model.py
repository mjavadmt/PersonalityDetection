import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import gc
from tqdm import tqdm
from pathlib import Path
import random

root_dir = Path(__file__).resolve().parents[2]
models_dir = root_dir / "models"


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 1000)
        self.dropout_1 = nn.Dropout(0.3)
        self.fc_2 = nn.Linear(1000, 200)
        self.dropout_2 = nn.Dropout(0.3)
        self.fc_3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.dropout_1(self.fc_1(x)))
        x = F.relu(self.dropout_2(self.fc_2(x)))
        x = self.fc_3(x)
        return F.sigmoid(x)


# Set the random seed for reproducibility
torch.manual_seed(42)


def train(X_data, y_data, model_name, input_dim=768):
    # Split the data into train, test, and validation sets
    X_train, y_train = X_data[0], y_data[0]
    X_test, y_test = X_data[1], y_data[1]
    X_val, y_val = X_data[2], y_data[2]
    # Create the model
    model = BinaryClassifier(input_dim).cuda()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 60
    batch_size = 32
    epochs_loss = {"train": [], "val": [], "test": []}
    epochs_acc = {"train": [], "val": [], "test": []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        indices = list(range(len(X_train)))
        random.shuffle(indices)
        X_train_shuffled = torch.stack([X_train[i] for i in indices])
        y_train_shuffled = torch.stack([y_train[i] for i in indices])
        for i in range(0, len(X_train_shuffled), batch_size):
            inputs = X_train_shuffled[i:i + batch_size]
            labels = y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()

            outputs = model(inputs).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss}")

        with torch.no_grad():
            train_outputs = model(X_train).float()
            train_loss = criterion(train_outputs, y_train)
            train_predicted_labels = train_outputs.round()
            train_accuracy = (y_train == train_predicted_labels).float().mean()
            epochs_loss["train"].append(train_loss.to("cpu").item())
            epochs_acc["train"].append(train_accuracy.to("cpu").item())

            val_outputs = model(X_val).float()
            val_loss = criterion(val_outputs, y_val)
            val_predicted_labels = val_outputs.round()
            val_accuracy = (y_val == val_predicted_labels).float().mean()
            epochs_loss["val"].append(val_loss.to("cpu").item())
            epochs_acc["val"].append(val_accuracy.to("cpu").item())

            test_outputs = model(X_test).float()
            test_loss = criterion(test_outputs, y_test)
            test_predicted_labels = test_outputs.round()
            test_accuracy = (y_test == test_predicted_labels).float().mean()
            epochs_loss["test"].append(test_loss.to("cpu").item())
            epochs_acc["test"].append(test_accuracy.to("cpu").item())

    torch.save(model.state_dict(), f"{models_dir.as_posix()}/feat_engineering_{model_name}.pth")



    print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return epochs_loss, epochs_acc
