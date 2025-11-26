import numpy as np
from torchvision import datasets, transforms
import os
from PIL import Image
import csv
from torch.utils.data import Dataset


data_dir = os.path.join(os.environ['HOME'], "PycharmProjects/FCIL-gan/datasets")
# print(data_dir)



class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iMNIST(iData):
    use_path = False  # Indicate whether to use paths or in-memory data
    train_trsf = [
        transforms.RandomCrop(28, padding=4),  # Random cropping with padding for MNIST
        transforms.RandomRotation(15),  # Random rotation for augmentation
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,), std=(0.3081,)
        ),  # Normalization for MNIST dataset
    ]

    class_order = np.arange(10).tolist()  # Classes from 0 to 9

    def download_data(self):
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data.numpy(), np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data.numpy(), np.array(
            test_dataset.targets
        )

class iFMNIST(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))  # Fashion-MNIST 官方统计
    ]
    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.FashionMNIST(data_dir, train=True,  download=True)
        test_dataset  = datasets.FashionMNIST(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data.numpy(), np.array(train_dataset.targets)
        self.test_data,  self.test_targets  = test_dataset.data.numpy(),  np.array(test_dataset.targets)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=False)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class TinyImageNet200(iData):
    use_path = True  # 保留路径，按原始 _setup_data 逻辑
    train_trsf = [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(64),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
        val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val")

        # 训练集
        train_dset = datasets.ImageFolder(train_dir)
        self.train_data = [path for path, _ in train_dset.imgs]  # 只保存路径
        self.train_targets = np.array([label for _, label in train_dset.imgs])

        # 类名到索引映射
        wnids_to_idx = train_dset.class_to_idx

        # 验证集
        self.test_data, self.test_targets = self._load_val_images(val_dir, wnids_to_idx)

    def _load_val_images(self, val_dir, wnids_to_idx):
        import csv
        images, labels = [], []
        anno_file = os.path.join(val_dir, "val_annotations.txt")
        with open(anno_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            mapping = {row[0]: row[1] for row in reader}  # img_name -> class_name

        img_dir = os.path.join(val_dir, "images")
        for img_name, class_name in mapping.items():
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                images.append(img_path)  # 保存路径
                labels.append(wnids_to_idx[class_name])
        return images, np.array(labels)
