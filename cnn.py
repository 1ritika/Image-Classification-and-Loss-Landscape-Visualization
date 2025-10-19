# version9.py - cnn model for cifar10 with scheduler and progress bars

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# import get_data and get_device from common/utils.py
from common.utils import get_data, get_device
from common.logutils import init_log_file, append_log

# ------------------------------
# model definition
# ------------------------------
class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.relu(x)
        x = self.bn3(x)
        x = F.max_pool2d(self.conv4(x), 2)
        x = F.relu(x)
        x = self.bn4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------
# training and evaluation routines
# ------------------------------
def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer) -> tuple[float, float]:
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = F.cross_entropy(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        total_correct += (outputs.argmax(dim=1) == target).sum().item()
        total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model: nn.Module, data_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            total_loss += loss.item() * data.size(0)
            total_correct += (outputs.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(model: nn.Module, data_loader: DataLoader,test_loader:DataLoader, optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler, epochs: int = 10) -> None:
    device = get_device()
    model.to(device)
    log_file = f"logs/cnn_log.csv"  # e.g., model_name = 'ResNet20'
    init_log_file(log_file, ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    # Evaluate only test set at epoch 0 to save memory
    train_loss,train_acc=evaluate(model,data_loader)
    test_loss, test_acc = evaluate(model, test_loader)
    append_log(log_file, [0, train_loss, train_acc, test_loss, test_acc])
    print(f"Epoch 0 | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("Training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, data_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)  
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"                     |  Test Loss: {test_loss:.4f} |  Test Acc: {test_acc:.4f}")

        append_log(log_file, [epoch+1, train_loss, train_acc, test_loss, test_acc])
    print("Training complete!")

# ------------------------------
# main function
# ------------------------------
def main() -> None:
    train_loader, test_loader = get_data('cifar10', batch_size=32)

    model = Net()
    print("Model Parameter Count:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


    train(model, train_loader, test_loader, optimizer, scheduler, epochs=50)  # train for 50 epochs
    

    test_loss, test_acc = evaluate(model, test_loader)
    test_error = 100 - (test_acc * 100)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc*100:.2f}% | Final Test Error: {test_error:.2f}%")


    train_loss, train_acc = evaluate(model, train_loader)
    train_error = 100 - (train_acc * 100)
    print(f"Final Train Loss: {train_loss:.4f} | Final Train Acc: {train_acc*100:.2f}% | Final Train Error: {train_error:.2f}%")


    # save model checkpoint
    torch.save(model.state_dict(), "models/cnn_cifar10.pth")
    print("Model saved as cnn_cifar10.pth")


if __name__ == '__main__':
    main()
