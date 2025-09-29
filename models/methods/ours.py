import copy
import math
import os
import random
from abc import ABC

import torch
import wandb
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn, optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import utils.losses as ls

from models.base import BaseLearner
from models.generator import Generator
from utils import util
from utils.data_manager import setup_seed, partition_data, DatasetSplit
from utils.net import IncrementalNet
from utils.util import cumulative, DeepInversionHook, save_image_batch

# cifar10, cifar100
data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**dict(data_normalize)),
])

# mnist
# data_normalize = dict(mean=(0.1307,), std=(0.3081,))
# train_transform = transforms.Compose([
#         transforms.RandomCrop(28, padding=4),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(**dict(data_normalize)),
# ])

# TinyImageNet
# data_normalize = dict(mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262))
# train_transform = transforms.Compose([
#     transforms.RandomCrop(64, padding=8),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(**dict(data_normalize)),
# ])
lamda = 1000
fishermax = 0.0001
a = 2
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


normalizer = Normalizer(**dict(data_normalize))


def _collect_all_images(root, nums=None, categories=None, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    """
    Collects images from the specified root directory, based on categories and a limit for the number of images per category.

    Args:
        nums (int): The number of images to collect per category.
        root (str): The root directory where categories are stored in separate subdirectories.
        categories (list): A list of categories (folder names) to collect images from.
        postfix (list): A list of file extensions to consider as valid image files (e.g., ['png', 'jpg']).

    Returns:
        list: A list of file paths for the images.
    """
    images = []
    labels = []

    # If postfix is a string, convert it to a list for uniformity
    if isinstance(postfix, str):
        postfix = [postfix]

    if categories is None:
        categories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    category_to_label = {category: idx for idx, category in enumerate(categories)}

    # Loop through each category in the categories list
    for category in categories:
        category_path = os.path.join(root, str(category))  # Create the full path for the category folder

        if os.path.isdir(category_path):  # Ensure the category folder exists
            files = os.listdir(category_path)  # Get all files in the category folder
            files = [f for f in files if any(f.endswith(pos) for pos in postfix)]  # Filter files by extension

            if nums is not None:
                random.shuffle(files)
                files = files[:nums]  # Limit the number of images

            # Add the full path of the files to the images list
            for f in files:
                images.append(os.path.join(category_path, f))
                labels.append(category_to_label[category])

    return images, labels


class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes=None, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images, self.labels = _collect_all_images(self.root, nums=nums, categories=classes)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
        self.root, len(self), self.transform)

    def delete_image(self, idx):
        if idx < 0 or idx >= len(self.images):
            raise IndexError("Index out of range")
        # Remove the image file
        image_path = self.images[idx]
        os.remove(image_path)
        self.images.pop(idx)
        self.labels.pop(idx)


class FCL_SSMR(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.prev_output = None
        self.fisher_list = {}
        self.mean_list = {}
        self.intersection_indices = []
        self.acc = []
        self.replay_task = []
        self.cur_classes = []
        self.all_client_features = {}
        self.client_prototypes = {}

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self.eval_network = None
        self.acc = []
        if self.replay_task:
            self.replay_task = list(set(self.replay_task + self.cur_classes))
        else:
            self.replay_task = list(set(self.cur_classes))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        is_error_task = True if self._cur_task < self._err_tasks else False
        self.cur_classes = data_manager.get_task(self._cur_task)
        test_indices = data_manager.get_test_classes(self._cur_task)
        print("current category: ", self.cur_classes)
        print("current test category: ", test_indices)
        if self._cur_task > 0:
            prev_classes = data_manager.get_task(self._cur_task - 1)
            self.intersection_indices = list(set(self.cur_classes).intersection(set(prev_classes)))

        print("fc_num", max(test_indices) + 1)
        self._network.update_fc(max(test_indices) + 1)
        self._total_classes = max(test_indices) + 1

        train_dataset = data_manager.get_dataset_with_split(  # * get the data for each task
            np.array(self.cur_classes),
            source="train",
            mode="train",
            task_error=is_error_task
        )

        test_dataset = data_manager.get_dataset_with_split(
            np.array(test_indices), source="test", mode="test", task_error=is_error_task,
            m_rate=self.args["m_rate"], categorys=self.cur_classes
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=True
        )
        # print(modified_indices_train)
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)
        # print("For task: {}, acc list max: {}".format(self._cur_task, self.acc))

        if self._cur_task + 1 != self.tasks:
            self.data_generation()

    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        cls_acc_list = []
        user_groups = partition_data(
            train_dataset.labels,
            beta=self.args["beta"],
            n_parties=self.args["num_users"]
        )
        test_acc_max = 0

        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            save_feature = True if com == self.args["com_round"] - 1 else False
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)

            if 0 < self._cur_task and com == 0:
                for idx in idxs_users:
                    local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4,
                                                    drop_last=True)
                    self._pre_train(self._old_network, local_train_loader, idx)

                syn_dataset, samples = self.feature_selection(copy.deepcopy(self._network), 0.2)
                print("delete label\n", samples["labels"])
                self.delete_selected_images(syn_dataset, samples)

            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                                batch_size=self.args["local_bs"], shuffle=True, num_workers=4,
                                                drop_last=True)

                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, idx, save_feature)
                else:
                    w, total_syn, total_local = self._local_finetune(self._old_network, copy.deepcopy(self._network),
                                                                     local_train_loader, idx, com, save_feature)
                    if com == 0 and self._cur_task != 0:
                        print(
                            "\t \t Client {},local dataset size {}, syntheic dataset size {} ".format(idx, total_local,
                                                                                                      total_syn))
                local_weights.append(copy.deepcopy(w))

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)  # global
            torch.save(self.all_client_features, "data/features.pt")

            if com % 1 == 0:
                cls_acc = self.per_cls_acc(self.test_loader, self._network)
                cls_acc_list.append(cls_acc)

                test_acc = self._compute_accuracy(self._network, test_loader)

                if test_acc > test_acc_max:
                    test_acc_max = test_acc
                    self.eval_network = copy.deepcopy(self._network)

                info = ("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc, ))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})
            
            max_len = max(len(i) for i in cls_acc_list)
            cls_acc_list = [np.pad(i, (0, max_len - len(i)), 'constant') for i in cls_acc_list]
            acc_arr = np.array(cls_acc_list)
            acc_max = acc_arr.max(axis=0)
            # print(np.mean(acc_max))
            # if self._cur_task == 3:
            #     acc_max = self.per_cls_acc(self.test_loader, self._network)
            # print("For task: {}, acc list max: {}".format(self._cur_task, acc_max))
            self.acc.append(np.mean(acc_max))

        print("Max test accuracy achieved: {:.2f}".format(test_acc_max))

    def _local_update(self, model, train_data_loader, client_id, save_feature=False):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            for idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # print(f"Max label: {labels.max().item()}, Min label: {labels.min().item()}")
                output = model(images)["logits"]
                self.prev_output = output
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if save_feature:
            self.save_features(model, train_data_loader, client_id)
        return model.state_dict()

    def _local_finetune(self, teacher, model, train_data_loader, client_id, com, save_feature=False):
        model.train()
        teacher.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        if com == 0:
            replay_task = [i for i in self.replay_task if i not in self.intersection_indices]
            self.syn_data_loader = self.get_syn_data_loader(classes=replay_task)
            # self.syn_data_loader = self.get_syn_data_loader()   # not select

        for it in range(self.args["local_ep"]):
            iter_loader = enumerate(zip(train_data_loader, self.syn_data_loader))
            total_local = 0.0
            total_syn = 0.0
            for idx, ((_, images, labels), (syn_input, _)) in iter_loader:
                images, labels, syn_input = images.cuda(), labels.cuda(), syn_input.cuda()
                output = model(images)["logits"]
                self.prev_output = output
                # for new tasks
                loss_ce = F.cross_entropy(output, labels)
                
                s_out = model(syn_input)["logits"]
                with torch.no_grad():
                    t_out = teacher(syn_input.detach())["logits"]
                    total_syn += syn_input.shape[0]
                    total_local += images.shape[0]
                # for old task
                loss_kd = _KD_loss(
                    s_out[:, self.replay_task],  # logits on previous tasks
                    t_out.detach(), 2
                    ) + _KD_loss(
                        output[:, self.replay_task],
                        teacher(images)["logits"], 2
                    )

                loss = loss_ce + self.args["kd"] * loss_kd
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=2.0)
                optimizer.step()

        if save_feature:
            self.save_features(model, train_data_loader, client_id)
        return model.state_dict(), total_syn, total_local

    def data_generation(self):
        nz = 256
        if self.args["dataset"] in ["cifar10", "cifar100"]:
            img_size = 32
            img_shape = (3, 32, 32)
            nc = 3
        elif self.args["dataset"] == "imagenet100":
            img_size = 128
            img_shape = (3, 128, 128)
            nc = 3
        elif self.args["dataset"] == "tiny_imagenet":
            img_size = 128
            img_shape = (3, 64, 64)
            nc = 3
        elif self.args["dataset"] == "mnist":
            img_size = 28
            img_shape = (1, 28, 28)
            nc = 1
        else:
            raise ValueError(f"Unsupported dataset: {self.args['dataset']}")
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        student = copy.deepcopy(self._network)
        # student.apply(weight_init)
        save_dir = self.args["imgs_path"]
        synthesizer = GlobalSynthesizer(copy.deepcopy(self._network), generator,
                                        nz=nz, num_classes=self._total_classes, img_size=img_shape,
                                        save_dir=save_dir, transform=train_transform,
                                        normalizer=normalizer, synthesis_batch_size=256,
                                        sample_batch_size=256, iterations=self.args["g_rounds"],
                                        warmup=20, lr_g=0.002, lr_z=0.01,
                                        args=self.args)
        # for it in range(1):
        synthesizer.synthesize(300 * self._cur_task, target=self.cur_classes)
        print("For task {}, data generation completed! ".format(self._cur_task))

    def _pre_train(self, teacher, train_data_loader, client_id):  # for every client
        teacher.eval()
        class_features = defaultdict(list)
        for batch_idx, (_, images, labels) in enumerate(train_data_loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                h = teacher(images)["features"]
                for i, label in enumerate(labels):
                    class_features[label.item()].append(h[i].cpu().numpy())

        cl_prototypes = {}
        for class_id, features in class_features.items():
            features = torch.tensor(features)
            cl_prototypes[class_id] = features.mean(dim=0)

        # save every client prototypes
        self.save_cl_prototypes(cl_prototypes, client_id)

    def per_cls_acc(self, val_loader, model):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i, (_, input, target) in enumerate(val_loader):
                input, target = input.cuda(), target.cuda()
                # compute output
                output = model(input)["logits"]
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)

        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)

        cls_acc = np.where(cls_cnt == 0, 0, cls_hit / cls_cnt)

        # print Per Class Accuracy
        # out_cls_acc = 'Per Class Accuracy: %s' % (
        #     np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))
        # print("----------------------------------------------")
        # print(out_cls_acc)
        return cls_acc

    def save_features(self, model, dataloader, client_id):
        model.eval()
        features = None
        classes = None

        with torch.no_grad():
            for i, (_, imgs, labels) in enumerate(dataloader):
                imgs, labels = imgs.cuda(), labels.cuda()
                h = model(imgs)["features"]

                if features is None:
                    features = h.detach().cpu()
                    classes = labels.detach().cpu()
                else:
                    features = torch.cat([features, h.detach().cpu()], dim=0)
                    classes = torch.cat([classes, labels.detach().cpu()], dim=0)

        values, indices = torch.sort(classes)
        bin_count = torch.bincount(classes).tolist()

        bin_count = [i for i in bin_count if i != 0]
        bin_count.insert(0, 0)
        bin_count_cum = cumulative(bin_count)

        clss = [int(classes[indices[bin_count_cum[a]]]) for a in range(len(bin_count) - 1)]

        mean = torch.empty((len(clss), features.shape[1]))
        var = torch.empty((len(clss), features.shape[1]))

        for a in range(len(bin_count) - 1):
            mean[a] = torch.mean(
                features[indices[bin_count_cum[a]: bin_count_cum[a + 1]]], dim=0
            )

            var[a] = torch.var(
                features[indices[bin_count_cum[a]: bin_count_cum[a + 1]]], dim=0
            )

        features_dict = {"mean": mean, "var": var, "labels": clss}
        self.all_client_features[client_id] = features_dict

    def save_cl_prototypes(self, cl_prototypes, client_id):
        if not hasattr(self, 'client_prototypes'):
            self.client_prototypes = {}

        self.client_prototypes[client_id] = cl_prototypes

        torch.save(self.client_prototypes, "data/client_prototypes.pth")

    def delete_selected_images(self, syn_dataset, samples):
        indices = samples["indices"]

        # sort
        indices = sorted(indices, reverse=True)

        for idx in indices:
            syn_dataset.delete_image(idx)
        print(f"Updated dataset size: {len(syn_dataset)}")

    def get_syn_data_loader(self, nums=800, classes=None):
        if self.args["dataset"] in ["cifar10", "cifar100"]:
            dataset_size = 60000
        elif self.args["dataset"] == "mnist":
            dataset_size = 50000
        elif self.args["dataset"] == "tiny_imagenet":
            dataset_size = 100000
        elif self.args["dataset"] == "imagenet100":
            dataset_size = 130000
        iters = math.ceil(dataset_size / (self.args["num_users"] * self.args["tasks"] * self.args["local_bs"]))
        # syn_bs = int(self._images_per_class * len(self.replay_task) / iters)
        syn_bs = int(8000 / iters)
        data_dir = self.save_dir
        # print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))

        syn_dataset = LabeledImageDataset(data_dir, classes=classes, transform=train_transform, nums=nums)
        print(f"syn_dataset size: {len(syn_dataset)}")
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True,
            num_workers=4, pin_memory=True, )
        return syn_data_loader

    def feature_selection(self, model, threshold=0.3):
        syn_data_loader = self.get_syn_data_loader()

        # Calculate the class prototypes (mean feature vectors) for each class
        class_prototypes = self.get_class_prototypes()

        # Prepare to store the selected samples
        selected_samples = {
            "indices": [],
            "labels": [],
            "deviations": []
        }
        all_deviations = []

        # Iterate through the synthetic data and calculate feature similarities
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(syn_data_loader):
                images, labels = images.cuda(), labels.cuda()

                # 1: Extract features from the synthetic data
                with torch.no_grad():
                    features = model(images)["features"]

                # 2: For each class, calculate the cosine similarity with the class prototype
                for i in range(features.size(0)):
                    feature = features[i].unsqueeze(0)  # (1, feature_size)
                    label = labels[i].item()
                    prototype = class_prototypes.get(label, None)

                    # Get the class prototype (mean feature) for the current label
                    if prototype is None:
                        continue
                    prototype = prototype.unsqueeze(0)

                    # Calculate cosine similarity between the generated feature and the class prototype
                    similarity = cosine_similarity(feature.cpu().numpy(), prototype.cpu().numpy())
                    similarity = (similarity + 1) / 2
                    similarity = torch.tensor(similarity, dtype=torch.float32)
                    deviation = -torch.log(similarity + 1e-8)  # Convert similarity to deviation (lower is better)
                    # print(f"{batch_idx}  {label}  {deviation.item()}")

                    # all_deviations.append(deviation.item())
                    # if deviation > threshold:
                    #     selected_samples["indices"].append(batch_idx * syn_data_loader.batch_size + i)
                    #     selected_samples["labels"].append(label)
                    #     selected_samples["deviations"].append(deviation.item())

                    # Temporarily store the sample info
                    selected_samples["indices"].append(batch_idx * syn_data_loader.batch_size + i)
                    selected_samples["labels"].append(label)
                    selected_samples["deviations"].append(deviation.item())

        if all_deviations:
            min_deviation = min(all_deviations)
            max_deviation = max(all_deviations)
            range_deviation = max_deviation - min_deviation if max_deviation > min_deviation else 1.0

            # Apply normalization
            normalized_deviations = [
                (d - min_deviation) / range_deviation for d in selected_samples["deviations"]
            ]
            filtered_indices = []
            filtered_labels = []
            print(normalized_deviations)
            for idx, norm_deviation in enumerate(normalized_deviations):
                if norm_deviation > (1 - threshold):
                    filtered_indices.append(selected_samples["indices"][idx])
                    filtered_labels.append(selected_samples["labels"][idx])

            # Update selected_samples with filtered data
            selected_samples["indices"] = filtered_indices
            selected_samples["labels"] = filtered_labels

        return syn_data_loader.dataset, selected_samples

        # self.syn_data_loader = self.get_syn_data_loader()

    def get_class_prototypes(self):
        class_prototypes = {}

        # Collect prototypes from each client
        for client_id, client_prototypes in self.client_prototypes.items():
            for class_id, prototype in client_prototypes.items():
                if class_id not in class_prototypes:
                    class_prototypes[class_id] = []
                class_prototypes[class_id].append(prototype)

        # Aggregate prototypes (by averaging) for each class across all clients
        for class_id, prototypes in class_prototypes.items():
            if len(prototypes) > 1:
                # If there are multiple prototypes, take the mean
                class_prototypes[class_id] = torch.mean(torch.stack(prototypes), dim=0)  # (feature_size)
            else:
                # If only one prototype exists, keep it as is
                class_prototypes[class_id] = prototypes[0]

        return class_prototypes



def reset_l0_fun(model):
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets):
        for img, target in zip(imgs, targets):

            target_dir = os.path.join(self.root, str(target.item()))
            os.makedirs(target_dir, exist_ok=True)

            save_path = os.path.join(target_dir, f"{self._idx}.png")
            save_image_batch(img.unsqueeze(0), save_path)
            self._idx += 1

    # def get_dataset(self, nums=None, transform=None, labeled=True):
    #     return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


class GlobalSynthesizer(ABC):
    def __init__(self, teacher, generator, nz, num_classes, img_size,
                 iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=1, bn=3, oh=1,
                 save_dir='', transform=None,
                 normalizer=None, lr_z=0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0.9,
                 is_maml=1, args=None):
        super(GlobalSynthesizer, self).__init__()
        self.teacher = teacher
        # self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.ismaml = is_maml
        self.args = args

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.class_embedding = nn.Embedding(self.num_classes, self.nz).cuda()  # Add class embedding

        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])

        if "cifar" in self.args["dataset"]:
            self.aug = transforms.Compose([
                transforms.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                transforms.RandomHorizontalFlip(),
                normalizer,
            ])
        elif "mnist" in self.args["dataset"]:
            self.aug = transforms.Compose([
                transforms.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                # transforms.ToTensor(),
                normalizer,
            ])

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, g_rounds, target=None):
        self.ep += 1
        # self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)

        # best_inputs = None
        best_inputs_list = []
        best_targets_list = []
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()
        z.requires_grad = True

        # If targets are given, generate random classes
        possible_targets = torch.tensor(target)
        targets = possible_targets[torch.randint(0, len(possible_targets), (self.synthesis_batch_size,))]
        targets = targets.cuda()

        # Get class embeddings for targets
        # class_embeddings = self.class_embedding(targets)

        fast_generator = self.generator.copy()
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        rounds = self.args["g_rounds"] + g_rounds
        prog_bar = tqdm(range(rounds))
        for _, it in enumerate(prog_bar):
            # Pass both noise z and class embeddings to generator
            inputs = fast_generator(z, targets)  # Pass embeddings here
            inputs_aug = self.aug(inputs)
            t_out = self.teacher(inputs_aug)["logits"]
            fake_h = self.teacher(inputs_aug)["features"]

            # if targets is None:
            #     targets = torch.argmax(t_out, dim=-1)
            #     targets = targets.cuda()

            loss_oh = F.cross_entropy(t_out, targets)
            loss_bn = sum([h.r_feature for h in self.hooks])
            smoothing = util.Gaussiansmoothing(channels=3, kernel_size=3)
            features_dict = torch.load("data/features.pt")
            loss_div = -ls.js_divergence(inputs_aug, self.synthesis_batch_size // 3)

            sigma_1, mu1 = torch.var_mean(fake_h, dim=0, unbiased=False)
            features_loss = calculate_features_loss(features_dict, targets, sigma_1, mu1)

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_div + features_loss
            with torch.no_grad():

                if loss.item() < best_cost or len(best_inputs_list) < a:
                    best_cost = loss.item()
                    best_inputs = inputs.data.cpu()
                    best_inputs_list.append(best_inputs.clone())
                    best_targets_list.append(targets.clone())

                    # rounds: a
                    if len(best_inputs_list) > a:
                        best_inputs_list.pop(0)
                        best_targets_list.pop(0)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(fast_generator.parameters(), max_norm=2.0)

            if self.ismaml:
                if it == 0:
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()

            optimizer.step()

            info = ("generator train: Epoch {}/{} =>  g_losses {:.2f}, div {:.2f}, class {:.2f},"
                    "features {:.2f}, loss_bn {:.2f}"
                    .format(it + 1, rounds, loss.item(), loss_div.item(), loss_oh.item(),
                            features_loss, loss_bn))
            prog_bar.set_description(info)

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        # self.student.train()
        self.prev_z = (z, targets)

        # self.data_pool.add(best_inputs, targets)
        # best a rounds add
        for best_inputs, best_targets in zip(best_inputs_list, best_targets_list):
            self.data_pool.add(best_inputs, best_targets)


def calculate_features_loss(features_dict, labels, sigma_1, mu1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_clients = len(features_dict)
    losses = []

    for client in range(num_clients):
        client_features = features_dict[client]
        sigma_2, mu2 = ls.merge_gaussians(client_features, labels)

        features_loss = torch.norm(mu1 - mu2.to(device)) + torch.norm(sigma_1 - sigma_2.to(device))
        losses.append(features_loss.item())

    max_loss_index = losses.index(max(losses))

    weighted_avg_loss = sum(losses) / num_clients

    return max(losses)


def _KD_loss(pred, soft, T):
    if pred.shape != soft.shape:
        num_c = min(pred.size(1), soft.size(1))
        pred = pred[:, :num_c]
        soft = soft[:, :num_c]
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
