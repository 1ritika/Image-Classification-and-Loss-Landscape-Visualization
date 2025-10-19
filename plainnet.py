import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import get_device, get_data
from common.logutils import init_log_file, append_log

def multiply_by_255(x):
    return x.mul(255)

# basic conv block without skip connection
class PlainBlock(nn.Module):
    def __init__(self, in_chann, chann, stride):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(chann)
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out

# backbone model for plainnet
class BaseNet(nn.Module):
    def __init__(self, Block, n):
        super(BaseNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.convs = self._make_layers(Block, n)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, 10)

    def _make_layers(self, Block, n):
        layers = []
        in_chann = 16
        chann = 16
        for i in range(3):
            for j in range(n):
                if i > 0 and j == 0:
                    in_chann = chann
                    chann *= 2
                    stride = 2
                else:
                    stride = 1
                layers.append(Block(in_chann, chann, stride))
                in_chann = chann
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.convs(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def PlainNet(n):
    return BaseNet(PlainBlock, n)

# one epoch training
def train_epoch(model, data_loader, optimizer):
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for data, target in tqdm(data_loader, desc="Training"):
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

# full test evaluation
def evaluate(model, data_loader):
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

# train for multiple epochs
def train(model, train_loader, optimizer, epochs=10):
    device = get_device()
    model.to(device)
    print("training...")
    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    print("training complete!")

# main training function
def main(depth):
    if (depth - 2) % 6 != 0:
        raise ValueError("depth must be of the form 6*n+2")
    n = (depth - 2) // 6
    print(f"building plainnet with overall depth {depth} (n = {n})")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
        transforms.ToTensor(),
        transforms.Lambda(multiply_by_255),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(multiply_by_255),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
    ])

    batch_size = 128
    num_workers = 2

    train_loader, _ = get_data('cifar10', batch_size=batch_size, transform=train_transform, num_workers=num_workers)
    _, test_loader = get_data('cifar10', batch_size=batch_size, transform=test_transform, num_workers=num_workers)

    device = get_device()
    model = PlainNet(n).to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1 if n != 18 else 0.0001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    total_iterations = 64000
    current_iteration = 0
    epoch = 0
    log_file = f"logs/plainnet{depth}_log.csv"
    init_log_file(log_file, ['iteration', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    # log before training
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        batch_acc = correct / batch_size
        batch_loss = loss.item()

        total_loss_test = 0.0
        total_correct_test = 0
        total_samples_test = 0
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            out_test = model(x_test)
            loss_test = criterion(out_test, y_test)
            total_loss_test += loss_test.item() * x_test.size(0)
            _, pred_test = torch.max(out_test, 1)
            total_correct_test += (pred_test == y_test).sum().item()
            total_samples_test += x_test.size(0)
        avg_test_loss = total_loss_test / total_samples_test
        avg_test_acc = total_correct_test / total_samples_test
        append_log(log_file, [0, batch_loss, batch_acc, avg_test_loss, avg_test_acc])
    model.train()

    # iteration-wise training
    while current_iteration < total_iterations:
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            if current_iteration >= total_iterations:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            current_iteration += 1

            # warmup for plainnet-110
            if n == 18 and current_iteration < 400:
                if learning_rate != 0.0001:
                    learning_rate = 0.0001
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                    print(f"warmup: setting learning rate to {learning_rate} at iteration {current_iteration}")
            if n == 18 and current_iteration == 400:
                learning_rate = 0.001
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                print(f"warmup complete at iteration {current_iteration}: learning rate set to {learning_rate}")

            if n != 18 and current_iteration in [32000, 48000]:
                learning_rate /= 10.0
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                print(f"learning rate adjusted to {learning_rate} at iteration {current_iteration}")

            if current_iteration % 100 == 0:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                batch_acc = correct / batch_size
                batch_loss = loss.item()
                total_loss_test = 0.0
                total_correct_test = 0
                total_samples_test = 0
                for x_test, y_test in test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    out_test = model(x_test)
                    loss_test = criterion(out_test, y_test)
                    total_loss_test += loss_test.item() * x_test.size(0)
                    _, pred_test = torch.max(out_test, 1)
                    total_correct_test += (pred_test == y_test).sum().item()
                    total_samples_test += x_test.size(0)
                avg_test_loss = total_loss_test / total_samples_test
                avg_test_acc = total_correct_test / total_samples_test
                print(f"iteration {current_iteration}, loss: {batch_loss:.4f}, batch accuracy: {batch_acc:.2f}%")
                append_log(log_file, [current_iteration, batch_loss, batch_acc, avg_test_loss, avg_test_acc])

        # epoch end evaluation
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)
        avg_test_loss = total_loss / total_samples
        test_accuracy = 100 * total_correct / total_samples
        test_error = 100 - test_accuracy
        print(f"epoch {epoch+1} completed. test loss: {avg_test_loss:.4f}, test accuracy: {test_accuracy:.2f}%, test error: {test_error:.2f}%")
        epoch += 1

    print("training completed.")

    # final train evaluation
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
    avg_train_loss = total_loss / total_samples
    train_accuracy = 100 * total_correct / total_samples
    train_error = 100 - train_accuracy
    print(f"final train loss: {avg_train_loss:.4f}, final train accuracy: {train_accuracy:.2f}%, final train error: {train_error:.2f}%")

    # final test evaluation
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
    avg_test_loss = total_loss / total_samples
    test_accuracy = 100 * total_correct / total_samples
    test_error = 100 - test_accuracy
    print(f"final test loss: {avg_test_loss:.4f}, final test accuracy: {test_accuracy:.2f}%, final test error: {test_error:.2f}%")

    # save model
    ckpt_name = f"models/plainnet-{depth}_cifar10.pth"
    torch.save(model.state_dict(), ckpt_name)
    print("model saved as", ckpt_name)

if __name__ == '__main__':
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(depth)
