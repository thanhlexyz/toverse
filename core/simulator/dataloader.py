import torchvision
import torch
import os

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
FASHION_MNIST_MEAN = (0.5,)
FASHION_MNIST_STD = (0.5,)

def create(args):
    if args.dataset == 'torch/cifar10':
        path = f'{args.data_dir}/cifar10.pt'
        transform     = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        need_rebuild = not os.path.exists(path)
        if not need_rebuild:
            data = torch.load(path, map_location='cpu', weights_only=False)
            need_rebuild = 'train' not in data
        if need_rebuild:
            cache_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=4)
            images, labels = next(iter(cache_loader))
            data = {'train': {'images': images.half(), 'labels': labels}}
            torch.save(data, path)
        return data, train_loader, test_loader
    elif args.dataset == 'fashion_mnist':
        transform     = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD)])
        train_dataset = torchvision.datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset  = torchvision.datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=transform)
        test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader
