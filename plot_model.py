import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
from visualizeLosses import generate_contour_plot
from common.utils import get_data

def multiply_by_255(x): return x.mul(255)

def get_model_and_loader(model_name, depth=None):
    model_name = model_name.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "cnn":
        from cnn import Net
        model = Net()
        path = "models/cnn_cifar10.pth"
        batch_size = 32

        # use default transform (no custom override)
        _, test_loader = get_data('cifar10', batch_size=batch_size, num_workers=2)

    elif model_name == "resnet":
        if depth is None:
            raise ValueError("Depth required for ResNet")
        from resnet import ResNet
        n = (depth - 2) // 6
        model = ResNet(n=n)
        path = f"models/resnet-{depth}_cifar10.pth"
        batch_size = 128

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(multiply_by_255),
            transforms.Normalize([125., 123., 114.], [1., 1., 1.])
        ])
        _, test_loader = get_data('cifar10', batch_size=batch_size, transform=transform, num_workers=2)

    elif model_name == "plainnet":
        if depth is None:
            raise ValueError("Depth required for PlainNet")
        from plainnet import PlainNet
        n = (depth - 2) // 6
        model = PlainNet(n=n)
        path = f"models/plainnet-{depth}_cifar10.pth"
        batch_size = 128

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(multiply_by_255),
            transforms.Normalize([125., 123., 114.], [1., 1., 1.])
        ])
        _, test_loader = get_data('cifar10', batch_size=batch_size, transform=transform, num_workers=2)

    else:
        raise ValueError("Unsupported model name. Use 'cnn', 'resnet', or 'plainnet'.")

    # Load model checkpoint
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model, test_loader, device

def main():
    if len(sys.argv) < 2:
        print("Usage: python plots.py [model_name] [depth (for resnet/plainnet)]")
        sys.exit(1)

    model_name = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 and model_name != "cnn" else None

    model, loader, device = get_model_and_loader(model_name, depth)

    title = f"{model_name.title()}-{depth} Loss Landscape" if depth else "CNN Loss Landscape"
    filename = f"plots/{model_name}{'-' + str(depth) if depth else ''}_contour.png"

    generate_contour_plot(
        model=model,
        data_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        title=title,
        filename=filename
    )

if __name__ == '__main__':
    main()
