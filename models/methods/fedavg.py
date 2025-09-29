import copy
import torch
import wandb
from sklearn.metrics import confusion_matrix
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from models.base import BaseLearner
from utils.data_manager import setup_seed, partition_data, DatasetSplit
from utils.net import IncrementalNet


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


class FedAvg(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.acc = []
        self.cur_classes = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self.acc = []

    def incremental_train(self, data_manager):
        self._cur_task += 1
        is_error_task = True if self._cur_task < self._err_tasks else False
        self.cur_classes = data_manager.get_task(self._cur_task)
        test_indices = data_manager.get_test_classes(self._cur_task)
        print("current category: ", self.cur_classes)
        print("current test category: ", test_indices)
        print("fc_num", max(test_indices) + 1)
        self._network.update_fc(max(test_indices) + 1)

        train_dataset = data_manager.get_dataset_with_split(  # * get the data for each task
            np.array(self.cur_classes),
            source="train",
            mode="train",
            task_error=is_error_task
        )

        test_dataset = data_manager.get_dataset_with_split(
            np.array(test_indices), source="test", mode="test", task_error=is_error_task,
            m_rate=1.0, categorys=self.cur_classes
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=True
        )
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)
        # print("For task: {}, acc list max: {}".format(self._cur_task, self.acc))

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
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                                batch_size=self.args["local_bs"], shuffle=True, num_workers=4)

                w = self._local_update(copy.deepcopy(self._network), local_train_loader)
                local_weights.append(copy.deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
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
            # acc_arr = np.array(cls_acc_list)
            # acc_max = acc_arr.max(axis=0)
            # print(np.mean(acc_max))
            # print("For task: {}, acc list max: {}".format(self._cur_task, acc_max))
            # self.acc.append(np.mean(acc_max))

        print("Max test accuracy achieved: {:.2f}".format(test_acc_max))

    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for it in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # print(f"Max label: {labels.max().item()}, Min label: {labels.min().item()}")
                output = model(images)["logits"]
                # print(f"Model Output Shape: {output.shape}")
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

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
        # unique_classes = np.unique(all_targets)
        # unique_pred_classes = np.unique(all_preds)
        # print("unique_classes:", unique_classes)
        # print("unique_pred_classes:", unique_pred_classes)

        cf = confusion_matrix(all_targets, all_preds).astype(float)

        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)

        cls_acc = cls_hit / cls_cnt

        # print accurate
        out_cls_acc = 'Per Class Accuracy: %s' % (
            np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))
        print("----------------------------------------------")
        print(out_cls_acc)
        return cls_acc

