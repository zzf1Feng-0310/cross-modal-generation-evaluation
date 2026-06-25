import os
import pickle
import numpy as np
from PIL import Image
import paddle
from paddle.io import Dataset, DataLoader, Subset
from paddle.vision.transforms import (
    Compose, RandomHorizontalFlip, RandomCrop, ColorJitter, ToTensor, Normalize
)
import matplotlib.pyplot as plt
import random

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

# ========== 通用 CIFAR-10 数据集==========
class CIFAR10Shared(Dataset):
    """先加载所有数据，允许传入不同的 transform"""
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.transform = transform
        self.data, self.labels = self._load_batches(root, train)

    def _load_batches(self, root, train):
        batches = [f'data_batch_{i}' for i in range(1,6)] if train else ['test_batch']
        all_data, all_labels = [], []
        for batch in batches:
            file_path = os.path.join(root, batch)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='bytes')
                all_data.append(entry[b'data'])
                all_labels.extend(entry[b'labels'])
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # N H W C
        return all_data, all_labels

#支持传入不同transform，提升加载效率（训练集/验证集公用一份数据）
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

# ========== 数据增强与归一化 ==========
train_transform = Compose([
    RandomHorizontalFlip(0.5),
    RandomCrop(32, padding=4),
    ColorJitter(0.2, 0.2, 0.2),
    ToTensor(),
    Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

test_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# ========== 获取 DataLoader ==========
#get_dataloaders函数按 20% 比例拆分训练 / 验证集
def get_dataloaders(data_root, batch_size=64, val_ratio=0.2, num_workers=0):
    set_seed(42)

    # 训练集完整数据（带增强）
    train_full = CIFAR10Shared(data_root, train=True, transform=train_transform)
    # 验证集使用同一个数据源，但应用测试 transform
    val_full = CIFAR10Shared(data_root, train=True, transform=test_transform)
    # 测试集
    test_set = CIFAR10Shared(data_root, train=False, transform=test_transform)

    total = len(train_full)
    val_size = int(total * val_ratio)
    indices = list(range(total))
    random.shuffle(indices)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader

# ========== 类别分布可视化 ==========
#检查数据集类别是否均衡，保证实验公平性
def plot_class_distribution(dataset, save_path='class_distribution.png'):
    labels = [dataset[i][1] for i in range(len(dataset))]
    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    plt.figure(figsize=(10,5))
    plt.hist(labels, bins=range(11), align='left', rwidth=0.8)
    plt.xticks(range(10), classes, rotation=45)
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"分布图已保存至 {save_path}")