import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from common.utils import get_device, get_data
from common.logutils import init_log_file, append_log

# scale pixel values
def multiply_by_255(x):
    return x.mul(255)

# basic residual block
class ResBlockA(nn.Module):
    def __init__(self, in_chann, chann, stride):
        super(ResBlockA, self).__init__()
        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(chann)
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        # skip connection
        if x.shape == y.shape:
            z = x
        else:
            z = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_channels = x.size(1)
            y_channels = y.size(1)
            ch_pad = (y_channels - x_channels) // 2
            z = F.pad(z, pad=(0, 0, 0, 0, ch_pad, ch_pad), mode="constant", value=0)
        z = z + y
        z = F.relu(z)
        return z

# resnet backbone
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
        # three stages: 32x32 → 16x16 → 8x8
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

def ResNet(n):
    return BaseNet(ResBlockA, n)

# main training function
def main(depth):
    if (depth - 2) % 6 != 0:
        raise ValueError("depth must be of the form 6*n+2")
    n = (depth - 2) // 6
    print(f"building resnet with overall depth {depth} (n = {n})")

    # data transforms
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

    print(f"Train dataset size: {len(train_loader.dataset)} images")
    print(f"Test dataset size: {len(test_loader.dataset)} images")

    device = get_device()
    model = ResNet(n).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    # warmup lr for resnet-110
    if n == 18:
        learning_rate = 0.01
        print("using warmup learning rate for resnet-110: starting with lr=0.01")
    else:
        learning_rate = 0.1

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    total_iterations = 64000
    current_iteration = 0
    epoch = 0

    log_file = f"logs/resnet{depth}_log.csv"
    init_log_file(log_file, ['iteration', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    # evaluate before training
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

    # training loop
    while current_iteration < total_iterations:
        model.train()
        for inputs, labels in train_loader:
            if current_iteration >= total_iterations:
                break

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_iteration += 1

            # warmup for first 400 steps
            if n == 18 and current_iteration < 400:
                if learning_rate != 0.01:
                    learning_rate = 0.01
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                    print(f"warmup: setting learning rate to {learning_rate} at iteration {current_iteration}")
            if n == 18 and current_iteration == 400:
                learning_rate = 0.1
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                print(f"warmup complete at iteration {current_iteration}: learning rate set to {learning_rate}")

            # lr decay
            if current_iteration in [32000, 48000]:
                learning_rate /= 10.0
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                print(f"learning rate adjusted to {learning_rate} at iteration {current_iteration}")

            # log every 100 iters
            if current_iteration % 100 == 0:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                batch_acc = correct / batch_size
                batch_loss = loss.item()

                model.eval()
                total_loss_test = 0.0
                total_correct_test = 0
                total_samples_test = 0
                with torch.no_grad():
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
                model.train()

                append_log(log_file, [current_iteration, batch_loss, batch_acc, avg_test_loss, avg_test_acc])
                print(f"iter {current_iteration}, train loss: {batch_loss:.4f}, acc: {batch_acc*100:.2f}%, test acc: {avg_test_acc*100:.2f}%")

        # show accuracy at end of epoch
        model.eval()
        total_loss_epoch = 0.0
        total_correct_epoch = 0
        total_samples_epoch = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss_epoch += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct_epoch += (predicted == labels).sum().item()
                total_samples_epoch += inputs.size(0)
        avg_test_loss = total_loss_epoch / total_samples_epoch
        test_accuracy = 100 * total_correct_epoch / total_samples_epoch
        test_error = 100 - test_accuracy
        print(f"epoch {epoch+1} completed. test loss: {avg_test_loss:.4f}, test accuracy: {test_accuracy:.2f}%, test error: {test_error:.2f}%")
        epoch += 1

    print("training completed.")

    # final train accuracy
    model.eval()
    total_loss_train = 0.0
    total_correct_train = 0
    total_samples_train = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == labels).sum().item()
            total_samples_train += inputs.size(0)
    avg_train_loss = total_loss_train / total_samples_train
    train_accuracy = 100 * total_correct_train / total_samples_train
    train_error = 100 - train_accuracy
    print(f"final train loss: {avg_train_loss:.4f}, final train accuracy: {train_accuracy:.2f}%, final train error: {train_error:.2f}%")

    # final test accuracy
    total_loss_test = 0.0
    total_correct_test = 0
    total_samples_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss_test += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct_test += (predicted == labels).sum().item()
            total_samples_test += inputs.size(0)
    avg_test_loss = total_loss_test / total_samples_test
    test_accuracy = 100 * total_correct_test / total_samples_test
    test_error = 100 - test_accuracy
    print(f"final test loss: {avg_test_loss:.4f}, final test accuracy: {test_accuracy:.2f}%, final test error: {test_error:.2f}%")

    # save model
    os.makedirs("models", exist_ok=True)
    ckpt_name = f"models/resnet-{depth}_cifar10.pth"
    torch.save(model.state_dict(), ckpt_name)
    print("model saved as", ckpt_name)

if __name__ == '__main__':
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(depth)
