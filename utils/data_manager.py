import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.backends import cudnn
from wandb.wandb_torch import torch
import torch, copy
import os, pdb, random
from utils.dataset import iCIFAR10, iCIFAR100, iMNIST, TinyImageNet200


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def partition_data(y_train, beta=0.4, n_parties=5):
    data_size = y_train.shape[0]
    net_dataidx_map = {i: [] for i in range(n_parties)}
    labels = np.unique(y_train)

    if beta == 0:  # for iid
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif beta > 0:  # for niid
        indices = np.arange(data_size)

        np.random.seed(402)
        np.random.shuffle(indices)

        correct_data_split = dirichlet_allocation(y_train, indices, n_parties, beta)
        for i in range(n_parties):
            net_dataidx_map[i].extend(np.array(indices)[correct_data_split[i]].tolist())

    return net_dataidx_map


def dirichlet_allocation(y_train, indices, n_clients, beta):
    idx_batch = [[] for _ in range(n_clients)]
    labels = np.unique(y_train[indices])
    min_size = 0
    min_require_size = 1

    while min_size < min_require_size:
        for k in labels:
            idx_k = np.where(y_train[indices] == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            proportions = np.array(
                [p * (len(idx_j) < len(indices) / n_clients) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])

    return idx_batch


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def get_increment_task(dataset, tasks, overlap_rate=None):
    if dataset == "cifar10" or dataset == "mnist":
        increment_task = [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
    elif dataset == "cifar100":
        if tasks == 5:
            if overlap_rate == 0.4:
                n_task, cls_per_task, share = 5, 20, 8
                increment_task = [list(range(t * (cls_per_task - share),
                                     t * (cls_per_task - share) + cls_per_task))
                          for t in range(n_task)]

        else:
            if overlap_rate == 0.4:
                n_task, cls_per_task, share = 10, 10, 4
                increment_task = [list(range(t * (cls_per_task - share),
                                     t * (cls_per_task - share) + cls_per_task))
                          for t in range(n_task)]
    
    elif dataset == "tiny_imagenet":
        n_task, cls_per_task, share = 10, 20, 8
        increment_task = [list(range(t * (cls_per_task - share),
                                     t * (cls_per_task - share) + cls_per_task))
                          for t in range(n_task)]

    return increment_task


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, tasks, overlap_rate):

        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        self._increment_task = get_increment_task(self.dataset_name, tasks, overlap_rate)
        self.overlap_rate = overlap_rate

    @property
    def nb_tasks(self):
        return len(self._increment_task)

    def get_task(self, task):
        return self._increment_task[task]

    def get_test_classes(self, task):
        if task < 0 or task >= len(self._increment_task):
            raise ValueError("cur is out of range")

        test_classes = []
        for i in range(task + 1):
            test_classes.extend(self._increment_task[i])

        if task > 0:
            test_classes = list(set(test_classes))

        return test_classes

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, task_error, appendent=None, m_rate=None, categorys=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        if source == "train":
            for idx in indices:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
                data.append(class_data)
                targets.append(class_targets)
        elif source == "test":
            for idx in indices:
                if idx in categorys:
                    class_data, class_targets = self._select(
                        x, y, low_range=idx, high_range=idx + 1
                    )
                else:
                    class_data, class_targets = self._select_rmm(
                        x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                    )
                data.append(class_data)
                targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)
        # data, targets = np.concatenate(data), np.concatenate(targets)

        modified_indices = []
        if source == "train" and task_error:  # error

            all_classes = np.unique(targets)
            num_to_modify = max(1, int(len(all_classes) * self.overlap_rate / 2))
            # num_to_modify = 1
            selected_classes = all_classes[-num_to_modify:]

            print(f"Selected classes to modify: {selected_classes}")
            num_overlap = int(len(all_classes) * self.overlap_rate)
            overlap_classes = all_classes[-num_overlap:]

            # setup_seed(410)
            for class_to_modify in selected_classes:
                modify_idx = np.where(targets == class_to_modify)[0]
                num_modify = int(len(modify_idx) * 0.5)
                np.random.seed(200)
                modify_idx_change = np.random.choice(modify_idx, size=num_modify, replace=False)

                possible_labels = list(set(overlap_classes) - {class_to_modify})
                new_label = np.random.choice(possible_labels)

                targets[modify_idx_change] = new_label
                print(f"{class_to_modify}-{new_label}")

        return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        print(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # x may list(path) 
        selected_x = [x[i] for i in idxes]
        selected_y = y[idxes]
        return selected_x, selected_y

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        if m_rate != 0:
            selected_idxes = np.random.randint(0, len(idxes), size=int(m_rate * len(idxes)))
            idxes = idxes[selected_idxes]
        idxes = np.sort(idxes)
        selected_x = [x[i] for i in idxes]
        selected_y = y[idxes]
        return selected_x, selected_y
        
    # def _select(self, x, y, low_range, high_range):
    #     idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
    #     return x[idxes], y[idxes]

    # def _select_rmm(self, x, y, low_range, high_range, m_rate):
    #     assert m_rate is not None
    #     if m_rate != 0:
    #         idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
    #         selected_idxes = np.random.randint(
    #             0, len(idxes), size=int(m_rate * len(idxes))
    #         )
    #         new_idxes = idxes[selected_idxes]
    #         new_idxes = np.sort(new_idxes)
    #     else:
    #         new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
    #     return x[new_idxes], y[new_idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx, image, label = self.dataset[self.idxs[item]]
        return idx, image, label


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "mnist":
        return iMNIST()
    elif name == "tiny_imagenet":
        return TinyImageNet200()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
